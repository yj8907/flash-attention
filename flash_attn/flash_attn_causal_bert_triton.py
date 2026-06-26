

import math

import torch
import triton
import triton.language as tl
import warnings


@triton.jit
def _load_k(
    k_ptrs,
    start_n,
    stride_kn,
    offs_d,
    offs_n,
    seqlen_k,
    headdim,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):

        # -- compute qk ----
    if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    return k

@triton.jit
def _load_q(
    q_ptrs,
    offs_d,
    offs_m,
    seqlen_q,
    headdim,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    return q

@triton.jit
def _load_v(
    v_ptrs,
    start_n,
    stride_vn,
    offs_d,
    offs_n,
    seqlen_k,
    headdim,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
        if EVEN_HEADDIM:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )
    return v


@triton.jit
def _load_bias(
    b_ptrs,
    start_n,
    offs_m,
    offs_n,
    seqlen_q,
    seqlen_k,
    BIAS_TYPE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if BIAS_TYPE == "vector":
            if EVEN_N:
                bias = tl.load(b_ptrs + start_n).to(tl.float32)
            else:
                bias = tl.load(
                    b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                ).to(tl.float32)
            bias = bias[None, :]
    elif BIAS_TYPE == "matrix":
        if EVEN_M & EVEN_N:
            bias = tl.load(b_ptrs + start_n).to(tl.float32)
        else:
            bias = tl.load(
                b_ptrs + start_n,
                mask=(offs_m[:, None] < seqlen_q)
                & ((start_n + offs_n)[None, :] < seqlen_k),
                other=0.0,
            ).to(tl.float32)
    
    return bias


@triton.jit
def _o_scale(
    acc_o,
    m_i,
    lse_i,
    t_ptrs,
):
    
    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    # tl.store(t_ptrs, o_scale)
    # o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    return acc_o



@triton.jit
def _fwd_kernel_inner(
    q, k, v, bias,
    acc_o,
    m_i,
    lse_i,
    start_n,
    t_ptrs,
    offs_m,
    offs_n,
    offs_d,
    headdim,
    softmax_scale,
    seqlen_k,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k.T)
    # Trying to combine the two masks seem to make the result wrong
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
    if IS_CAUSAL:
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
    if BIAS_TYPE != "none":
        # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
        # can then fuse the mult and add into an fma instruction. But if we have bias we need to
        # to multiply with softmax_scale here.
        qk = qk * softmax_scale + bias
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
    else:
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    # scale acc_o
    acc_o_scale = tl.exp(m_i - m_ij)

    # # -- update output accumulator --
    # BUG: have to store and immediately load
    # tl.store(t_ptrs, acc_o_scale)
    # acc_o_scale = tl.load(t_ptrs)
    acc_o = acc_o * acc_o_scale[:, None]
    # update acc_o
    p = p.to(v.dtype)

    acc_o = acc_o+tl.dot(p, v)

    # -- update statistics
    m_i = m_ij

    l_i_new = tl.exp(lse_i - m_ij) + l_ij
    lse_i = m_ij + tl.log(l_i_new)

    return acc_o, m_i, lse_i


def _fwd_kernel_lookahead(
    start_n, 
    start_m,
    k_ptrs,
    v_ptrs,
    t_ptrs,
    w_k,
    w_v,
    q_from_n,
    headdim,
    stride_kn,
    stride_vn,
    offs_n,
    offs_d,
    softmax_scale,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LA_BLOCK_N_SIZE: tl.constexpr,
): 
    
    # prologue
    lse_i_from_n = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    m_i_from_n = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    acc_o_from_n = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    offs_m_from_n = start_n + tl.arange(0, BLOCK_N) # column n as query m
    # q_ptrs_from_n = (
    #     Q + off_b * stride_qb + off_h * stride_qh + (offs_m_from_n[:, None] * stride_qm + offs_d[None, :])
    # )
    
    # q_from_n = _load_q(q_ptrs=q_ptrs_from_n, offs_d=offs_d, offs_m=offs_m_from_n, seqlen_q=seqlen_q, headdim=headdim,
    #     EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)

    # also read the block right before query m's block
    near_m_n = (tl.cdiv(start_m * BLOCK_M, BLOCK_N) - 1) * BLOCK_N
    near_m_n = tl.multiple_of(near_m_n, BLOCK_N)
    
    # iterate starts from block m. This way, it can collect as many historical tokens as possible
    for i in range(LA_BLOCK_N_SIZE):
        next_n = near_m_n - i * BLOCK_N

        if next_n >= start_n:
            k_from_n = _load_k(k_ptrs=k_ptrs, start_n=next_n, stride_kn=stride_kn, offs_d=offs_d, offs_n=offs_n, seqlen_k=start_m*BLOCK_M,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=tl.constexpr(False), EVEN_N=tl.constexpr(False))
            v_from_n = _load_v(v_ptrs=v_ptrs, start_n=next_n, stride_vn=stride_vn, offs_d=offs_d, offs_n=offs_n, seqlen_k=start_m*BLOCK_M,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=tl.constexpr(False), EVEN_N=tl.constexpr(False))

            acc_o_from_n, m_i_from_n, lse_i_from_n = \
                _fwd_kernel_inner(q=q_from_n, k=k_from_n, v=v_from_n, bias=None,
                acc_o=acc_o_from_n, m_i=m_i_from_n, lse_i=lse_i_from_n, 
                start_n=next_n, t_ptrs=t_ptrs,
                offs_m=offs_m_from_n, offs_n=offs_n, offs_d=offs_d, 
                headdim=headdim, softmax_scale=softmax_scale,
                seqlen_k=start_m*BLOCK_M,
                EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=tl.constexpr(False), EVEN_N=tl.constexpr(False),
                BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_N, BIAS_TYPE='none', IS_CAUSAL=tl.constexpr(False),
                )

    # when start_m  <= start_n, no look ahead iterations take place. therefore, m_i, lse_i are all infinite.
    if start_m  * BLOCK_M > start_n:
        acc_o_from_n = _o_scale(acc_o=acc_o_from_n, m_i=m_i_from_n, lse_i=lse_i_from_n, t_ptrs=t_ptrs) # (bN, d)

    delta_k = tl.dot(acc_o_from_n, w_k).to(tl.float16)   
    delta_v = tl.dot(acc_o_from_n, w_v).to(tl.float16)

    return delta_k, delta_v, 

# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_causal_bert(
    Q,
    K,
    V,
    Bias,
    Weight,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_wh,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LA_BLOCK_N_SIZE: tl.constexpr, # look ahead column block size (key, value)
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )

    # weight for key, value projection for lookahead
    w_ptrs = Weight + off_h * stride_wh
    offs_w_k = tl.arange(0, BLOCK_HEADDIM)[:, None] * 2*BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)[None, :]
    offs_w_v = tl.arange(0, BLOCK_HEADDIM)[:, None] * 2*BLOCK_HEADDIM + tl.arange(BLOCK_HEADDIM, 2*BLOCK_HEADDIM)[None, :]

    w_k = tl.load(w_ptrs + offs_w_k).to(tl.float32)
    w_v = tl.load(w_ptrs + offs_w_v).to(tl.float32)

    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    q = _load_q(q_ptrs=q_ptrs, offs_d=offs_d, offs_m=offs_m, seqlen_q=seqlen_q, headdim=headdim,
                EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)

    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        offs_m_from_n = start_n + tl.arange(0, BLOCK_N) # column n as query m
        q_ptrs_from_n = (
            Q + off_b * stride_qb + off_h * stride_qh + (offs_m_from_n[:, None] * stride_qm + offs_d[None, :])
        )
        
        q_from_n = _load_q(q_ptrs=q_ptrs_from_n, offs_d=offs_d, offs_m=offs_m_from_n, seqlen_q=seqlen_q, headdim=headdim,
            EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
        

        delta_k, delta_v = _fwd_kernel_lookahead(start_n=start_n, start_m=start_m, 
                                k_ptrs=k_ptrs, v_ptrs=v_ptrs, t_ptrs=t_ptrs, w_k=w_k, w_v=w_v,
                                q_from_n=q_from_n, headdim=headdim, stride_kn=stride_kn, stride_vn=stride_vn,
                                offs_n=offs_n, offs_d=offs_d, softmax_scale=softmax_scale,
                                BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_HEADDIM=EVEN_HEADDIM,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, LA_BLOCK_N_SIZE=LA_BLOCK_N_SIZE)


        # outer most loop attention
        k = _load_k(k_ptrs=k_ptrs, start_n=start_n, stride_kn=stride_kn, offs_d=offs_d, offs_n=offs_n, seqlen_k=seqlen_k,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
        k += delta_k
        k = k.to(tl.float16)

        v = _load_v(v_ptrs=v_ptrs, start_n=start_n, stride_vn=stride_vn, offs_d=offs_d, offs_n=offs_n, seqlen_k=seqlen_k,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_N, EVEN_N=EVEN_N)
        v += delta_v
        v = v.to(tl.float16)

        if BIAS_TYPE != "none":
            bias = _load_bias(b_ptrs=b_ptrs, start_n=start_n, offs_m=offs_m, offs_n=offs_n, 
                            seqlen_q=seqlen_q, seqlen_k=seqlen_k, BIAS_TYPE=BIAS_TYPE, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
        else:
            bias = None

        acc_o, m_i, lse_i = _fwd_kernel_inner(q=q, k=k, v=v, bias=bias,
                          acc_o=acc_o, m_i=m_i, lse_i=lse_i, start_n=start_n, 
                          t_ptrs=t_ptrs,
                          offs_m=offs_m, offs_n=offs_n, offs_d=offs_d, 
                          headdim=headdim, softmax_scale=softmax_scale, seqlen_k=seqlen_k,
                        EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL,
                          )

    acc_o = _o_scale(acc_o=acc_o, m_i=m_i, lse_i=lse_i, t_ptrs=t_ptrs)

    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )



def _flash_attn_causal_bert_forward(q, k, v, w, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    
    BLOCK_M = 16
    BLOCK_N = 64

    LA_BLOCK_N_SIZE = 2
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel_causal_bert[grid](
        q,
        k,
        v,
        bias,
        w,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        w.stride(0),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        LA_BLOCK_N_SIZE=LA_BLOCK_N_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


@triton.jit
def _bwd_kernel_inner(
    q,
    k,
    v,
    do,
    dk,
    dv,
    Di,
    dq_ptrs,
    bias,
    lse_i,
    offs_m_curr,
    offs_n,
    offs_d,
    headdim,
    seqlen_q,
    seqlen_k,
    softmax_scale,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    
    # recompute p = softmax(qk, dim=-1).T
    qk = tl.dot(q, k.T)
    # Trying to combine the two masks seem to make the result wrong
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
    if BIAS_TYPE != "none":
        qk = qk * softmax_scale + bias
    # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
    # Also wrong for headdim=64.
    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    if BIAS_TYPE == "none":
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
    else:
        p = tl.exp(qk - lse_i[:, None])
    # compute dv
    # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
    # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
    # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
    # the output is correct.

    # if EVEN_M:
    #     if EVEN_HEADDIM:
    #         do = tl.load(do_ptrs)
    #     else:
    #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    # else:
    #     if EVEN_HEADDIM:
    #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
    #     else:
    #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
    #                                    & (offs_d[None, :] < headdim), other=0.0)
    dv += tl.dot(p.to(do.dtype), do, trans_a=True)
    # compute dp = dot(v, do)
    # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
    # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
    # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()
    dp = tl.dot(do, v.T)

    # There's a race condition for headdim=48
    if not EVEN_HEADDIM:
        tl.debug_barrier()

    # compute ds = p * (dp - delta[:, None])
    # Putting the subtraction after the dp matmul (instead of before) is slightly faster

    # Converting ds to q.dtype here reduces register pressure and makes it much faster
    # for BLOCK_HEADDIM=128
    ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
    # compute dk = dot(ds.T, q)
    dk += tl.dot(ds, q, trans_a=True)
    # compute dq
    if not (
        EVEN_M & EVEN_HEADDIM
    ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
        tl.debug_barrier()
    if not ATOMIC_ADD:
        if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
            dq = tl.load(dq_ptrs, eviction_policy="evict_last")
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq, eviction_policy="evict_last")
        else:
            if EVEN_HEADDIM:
                dq = tl.load(
                    dq_ptrs,
                    mask=offs_m_curr[:, None] < seqlen_q,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                dq += tl.dot(ds, k)
                tl.store(
                    dq_ptrs,
                    dq,
                    mask=offs_m_curr[:, None] < seqlen_q,
                    eviction_policy="evict_last",
                )
            else:
                dq = tl.load(
                    dq_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_last",
                )
                dq += tl.dot(ds, k)
                tl.store(
                    dq_ptrs,
                    dq,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    eviction_policy="evict_last",
                )
    else:  # If we're parallelizing across the seqlen_k dimension
        dq = tl.dot(ds, k)
        if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
            tl.atomic_add(dq_ptrs, dq)
        else:
            if EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
            else:
                tl.atomic_add(
                    dq_ptrs,
                    dq,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                )

    return dk, dv

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    w_k,
    w_v,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LA_BLOCK_N_SIZE: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])

    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n + start_n * BLOCK_N
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + (start_n * BLOCK_N+offs_n)[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + ((start_n * BLOCK_N + offs_n)[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + ((start_n * BLOCK_N + offs_n)[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    k = _load_k(k_ptrs=k_ptrs, start_n=start_n, stride_kn=stride_kn, 
                offs_d=offs_d, offs_n=offs_n, seqlen_k=seqlen_k, headdim=headdim,
                EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
    v = _load_v(v_ptrs=v_ptrs, start_n=start_n, stride_vn=stride_vn, 
                offs_d=offs_d, offs_n=offs_n, seqlen_k=seqlen_k,
                headdim=headdim, 
                EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)


    offs_m_from_n = start_n + tl.arange(0, BLOCK_N) # column n as query m
    q_ptrs_from_n = (
            Q + (offs_m_from_n[:, None] * stride_qm + offs_d[None, :])
        )
    q_from_n = _load_q(q_ptrs=q_ptrs_from_n, offs_d=offs_d, offs_m=offs_m_from_n, seqlen_q=seqlen_q, headdim=headdim,
            EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
    

    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        delta_k, delta_v = _fwd_kernel_lookahead(start_n=start_n, start_m=start_m, 
                                    k_ptrs=k_ptrs, v_ptrs=v_ptrs, t_ptrs=None, w_k=w_k, w_v=w_v, 
                                    q_from_n=q_from_n, headdim=headdim, stride_kn=stride_kn, stride_vn=stride_vn,
                                    offs_n=offs_n, offs_d=offs_d, softmax_scale=softmax_scale,
                                    BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_HEADDIM=EVEN_HEADDIM,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, LA_BLOCK_N_SIZE=LA_BLOCK_N_SIZE)
        k += delta_k
        v += delta_v

        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)

        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        
        Di = tl.load(D + offs_m_curr)

        dk, dv = _bwd_kernel_inner(q=q, k=k, v=v, do=do, dk=dk, dv=dv,
                          Di=Di, dq_ptrs=dq_ptrs,
                          bias=bias, lse_i=lse_i,
                          offs_m_curr=offs_m_curr, offs_n=offs_n, offs_d=offs_d, 
                          headdim=headdim, seqlen_q=seqlen_q, seqlen_k=seqlen_k, softmax_scale=softmax_scale,
                          EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, 
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
                          BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, ATOMIC_ADD=ATOMIC_ADD
                          )

        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm

    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)
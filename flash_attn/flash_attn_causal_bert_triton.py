

import math

import torch
import triton
import triton.language as tl



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
def _load_weight(

)

@triton.jit
def _o_scale(
    acc_o,
    m_i,
    lse_i,
    t_ptrs,
):
    
    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
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
    qk += tl.dot(q, k, trans_b=True)
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
    tl.store(t_ptrs, acc_o_scale)
    acc_o_scale = tl.load(t_ptrs)
    acc_o = acc_o * acc_o_scale[:, None]
    # update acc_o
    p = p.to(v.dtype)

    tl.store(acc_o, acc_o+tl.dot(p, v))

    # -- update statistics
    m_i = m_ij
    tl.store(m_i, m_ij)

    l_i_new = tl.exp(lse_i - m_ij) + l_ij
    tl.store(lse_i, m_ij + tl.log(l_i_new))



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

    weight_ptrs = Weight + off_h * stride_wh
    offs_w = tl.arange(0, BLOCK_HEADDIM)[:, None] * 2*BLOCK_HEADDIM + tl.arange(0, 2*BLOCK_HEADDIM)[None, :]
    weight = tl.load(weight_ptrs + offs_w).to(tl.float32)

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
        
        # prologue
        lse_i_from_n = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        m_i_from_n = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        acc_o_from_n = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

        offs_m_from_n = start_n + tl.arange(0, BLOCK_N) # column n as query m
        q_ptrs_from_n = (
            Q + off_b * stride_qb + off_h * stride_qh + (offs_m_from_n[:, None] * stride_qm + offs_d[None, :])
        )
        
        q_from_n = _load_q(q_ptrs=q_ptrs_from_n, offs_d=offs_d, offs_m=offs_m_from_n, seqlen_q=seqlen_q, headdim=headdim,
            EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)

        # also read the block right before query m's block
        nearby_n = tl.cdiv(start_m * BLOCK_M, BLOCK_N) * BLOCK_N
        nearby_n = tl.multiple_of(nearby_n, BLOCK_N)
        
        for i in range(LA_BLOCK_N_SIZE+1):
            if i < LA_BLOCK_N_SIZE:
                next_n = start_n + i * BLOCK_N
            else:
                next_n = nearby_n

            k_from_n = _load_k(k_ptrs=k_ptrs, start_n=next_n, stride_kn=stride_kn, offs_d=offs_d, offs_n=offs_n, seqlen_k=start_m,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
            v_from_n = _load_v(v_ptrs=v_ptrs, start_n=next_n, stride_vn=stride_vn, offs_d=offs_d, offs_n=offs_n, seqlen_k=start_m,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_N, EVEN_N=EVEN_N)

            if BIAS_TYPE != "none":
                bias_from_n = _load_bias(b_ptrs=b_ptrs, start_n=next_n, offs_m=offs_m_from_n, offs_n=offs_n, 
                                seqlen_q=seqlen_q, seqlen_k=seqlen_k, BIAS_TYPE=BIAS_TYPE, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
            else:
                bias_from_n = None

            _fwd_kernel_inner(q=q_from_n, k=k_from_n, v=v_from_n, bias=bias_from_n,
                acc_o=acc_o_from_n, m_i=m_i_from_n, lse_i=lse_i_from_n, 
                start_n=start_n, t_ptrs=t_ptrs,
                offs_m=offs_m, offs_n=offs_n, offs_d=offs_d, 
                headdim=headdim, softmax_scale=softmax_scale,
                seqlen_k=seqlen_k,
                EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL,
                )

        acc_o_from_n = _o_scale(acc_o=acc_o_from_n, m_i=m_i_from_n, lse_i=lse_i_from_n, t_ptrs=t_ptrs) # (bN, d)
        delta_k = tl.dot(acc_o_from_n, weight[:, :BLOCK_HEADDIM])   
        delta_v = tl.dot(acc_o_from_n, weight[:, :BLOCK_HEADDIM])

        # outer most loop attention
        k = _load_k(k_ptrs=k_ptrs, start_n=start_n, stride_kn=stride_kn, offs_d=offs_d, offs_n=offs_n, seqlen_k=seqlen_k,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
        k += delta_k
        v = _load_v(v_ptrs=v_ptrs, start_n=start_n, stride_vn=stride_vn, offs_d=offs_d, offs_n=offs_n, seqlen_k=seqlen_k,
                    headdim=headdim, EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_N, EVEN_N=EVEN_N)
        v += delta_v
        
        if BIAS_TYPE != "none":
            bias = _load_bias(b_ptrs=b_ptrs, start_n=start_n, offs_m=offs_m, offs_n=offs_n, 
                            seqlen_q=seqlen_q, seqlen_k=seqlen_k, BIAS_TYPE=BIAS_TYPE, EVEN_M=EVEN_M, EVEN_N=EVEN_N)
        else:
            bias = None

        _fwd_kernel_inner(q=q, k=k, v=v, bias=bias,
                          acc_o=acc_o, m_i=m_i, lse_i=lse_i, start_n=start_n, 
                          k_ptrs=k_ptrs, v_ptrs=v_ptrs, b_ptrs=b_ptrs, t_ptrs=t_ptrs,
                          offs_m=offs_m, offs_n=offs_n, offs_d=offs_d, headdim=headdim, softmax_scale=softmax_scale,
                        stride_kn=stride_kn, stride_vn=stride_vn, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
                        EVEN_HEADDIM=EVEN_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL,
                          )

    acc_o = _o_scale(acc_o=acc_o, m_i=m_i, lse_i=lse_i, acc_o=acc_o, t_ptrs=t_ptrs)

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


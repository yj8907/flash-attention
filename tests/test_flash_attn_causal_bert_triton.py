
import math
import torch


def _inner(q, k, v, acc_o, m_i, lse_i, start_n, seqlen_k, offs_n, offs_m,
           softmax_scale, even_n, is_causal):
    """Literal port of `_fwd_kernel_inner` (BIAS_TYPE == 'none'), vectorized over [B,H]."""
    NEG = float("-inf")
    qk = torch.einsum("bhid,bhjd->bhij", q.float(), k.float())           # tl.dot(q, k.T), fp32 accum
    if not even_n:
        col = (start_n + offs_n)
        qk = qk + torch.where((col < seqlen_k)[None, None, None, :], 0.0, NEG)
    if is_causal:
        m = (offs_m[:, None] >= (start_n + offs_n)[None, :])
        qk = qk + torch.where(m[None, None], 0.0, NEG)
    m_ij = torch.maximum(qk.amax(dim=-1) * softmax_scale, lse_i)
    p = torch.exp(qk * softmax_scale - m_ij[..., None])
    l_ij = p.sum(dim=-1)
    acc_o_scale = torch.exp(m_i - m_ij)
    p16 = p.to(v.dtype)                                                  # p = p.to(v.dtype)
    acc_o = acc_o * acc_o_scale[..., None] + torch.einsum(
        "bhij,bhjd->bhid", p16.float(), v.float())
    m_i = m_ij
    l_i_new = torch.exp(lse_i - m_ij) + l_ij
    lse_i = m_ij + torch.log(l_i_new)
    return acc_o, m_i, lse_i


def _o_scale(acc_o, m_i, lse_i):
    return acc_o * torch.exp(m_i - lse_i)[..., None]


def causal_bert_torch(q, k, v, w, causal=False, softmax_scale=None,
                      block=64, la_block_n_size=2):
    """Faithful PyTorch equivalent of `_flash_attn_causal_bert_forward`.

    q,k,v : [B,S,H,D] (fp16/bf16, seqlen_q == seqlen_k == S, S % block == 0)
    w     : [H,D,2D]  -> w_k = w[h,:,:D], w_v = w[h,:,D:]
    returns o:[B,S,H,D] (same dtype as q), lse:[B,H,S_rounded]
    """
    B, S, H, D = q.shape
    assert w.shape == (H, D, 2 * D)
    BM = BN = block
    LA = la_block_n_size
    NEG = float("-inf")
    dev = q.device
    in_dtype = q.dtype
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    Q = q.permute(0, 2, 1, 3).contiguous()
    K = k.permute(0, 2, 1, 3).contiguous()
    V = v.permute(0, 2, 1, 3).contiguous()
    Wk = w[:, :, :D].float()
    Wv = w[:, :, D:].float()

    S_round = math.ceil(S / 128) * 128
    O = torch.empty(B, H, S, D, dtype=in_dtype, device=dev)
    LSE = torch.empty(B, H, S_round, dtype=torch.float32, device=dev)
    ar = torch.arange(BN, device=dev)
    offs_n = ar
    nM = (S + BM - 1) // BM

    def load_block(T, pos, seqlen_mask):
        out = torch.zeros(B, H, BN, D, dtype=T.dtype, device=dev)
        end = min(pos + BN, S)
        if 0 <= pos < S and end > pos:
            out[:, :, : end - pos, :] = T[:, :, pos:end, :]
        keep = (pos + ar) < seqlen_mask
        out = out * keep[None, None, :, None].to(out.dtype)
        return out

    for mb in range(nM):
        start_m = mb
        offs_m = mb * BM + ar
        q_main = Q[:, :, mb * BM:(mb + 1) * BM, :]
        end_n = S if not causal else min((mb + 1) * BM, S)

        lse_i = torch.full((B, H, BM), NEG, dtype=torch.float32, device=dev)
        m_i = torch.full((B, H, BM), NEG, dtype=torch.float32, device=dev)
        acc_o = torch.zeros(B, H, BM, D, dtype=torch.float32, device=dev)

        for start_n in range(0, end_n, BN):
            q_from_n = Q[:, :, start_n:start_n + BN, :]
            offs_m_from_n = start_n + ar
            lse_i_from_n = torch.full((B, H, BN), NEG, dtype=torch.float32, device=dev)
            m_i_from_n = torch.full((B, H, BN), NEG, dtype=torch.float32, device=dev)
            acc_o_from_n = torch.zeros(B, H, BN, D, dtype=torch.float32, device=dev)

            near_m_n = (((start_m * BM + BN - 1) // BN) - 1) * BN
            for i in range(LA):
                next_n = near_m_n - i * BN
                if next_n >= start_n:
                    k_from_n = load_block(K, next_n, seqlen_mask=mb * BM)
                    v_from_n = load_block(V, next_n, seqlen_mask=mb * BM)
                    acc_o_from_n, m_i_from_n, lse_i_from_n = _inner(
                        q_from_n, k_from_n, v_from_n, acc_o_from_n, m_i_from_n, lse_i_from_n,
                        start_n=next_n, seqlen_k=mb * BM, offs_n=offs_n, offs_m=offs_m_from_n,
                        softmax_scale=softmax_scale, even_n=False, is_causal=False)

            if start_m * BM > start_n:
                acc_o_from_n = _o_scale(acc_o_from_n, m_i_from_n, lse_i_from_n)

            delta_k = torch.einsum("bhid,hde->bhie", acc_o_from_n, Wk)
            delta_v = torch.einsum("bhid,hde->bhie", acc_o_from_n, Wv)

            k_blk = load_block(K, start_n, seqlen_mask=S).float() + delta_k
            v_blk = load_block(V, start_n, seqlen_mask=S).float() + delta_v
            k_blk = k_blk.to(torch.float16)
            v_blk = v_blk.to(torch.float16)

            acc_o, m_i, lse_i = _inner(
                q_main, k_blk, v_blk, acc_o, m_i, lse_i,
                start_n=start_n, seqlen_k=S, offs_n=offs_n, offs_m=offs_m,
                softmax_scale=softmax_scale, even_n=(S % BN == 0), is_causal=causal)

        acc_o = _o_scale(acc_o, m_i, lse_i)
        O[:, :, mb * BM:(mb + 1) * BM, :] = acc_o.to(in_dtype)
        LSE[:, :, mb * BM:(mb + 1) * BM] = lse_i

    return O.permute(0, 2, 1, 3).contiguous(), LSE
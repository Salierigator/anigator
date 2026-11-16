"""InfoNCE with logQ correction, temperature and per-anchor hard negatives.

- s(i,x) = (U_i . V_x) / tau, with U and V L2-normalized (cosine).
- logQ (x logq_alpha) applies to the in-batch logits, positive included, never to hard negatives.
- False negatives (two anchors with the same positive item) are masked to -inf so an anchor
  is not penalized for its own positive.
- Padded hard negatives are masked to -inf; beta scales the hard-neg branch of the
  denominator (adds log beta).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def info_nce_logq(U, V_pos, V_hn, hardneg_mask, pos_idx, logq, tau, beta, logq_alpha=1.0):
    """
    U          [B, d]      user vec (normalized)
    V_pos      [B, d]      positive item vec (normalized)
    V_hn       [B, m, d]   hard-neg item vec (normalized)
    hardneg_mask [B, m]    bool, True = real hard negative
    pos_idx    [B]         anime_idx of the positive (for logQ and the false-negative mask)
    logq       [num_items] dense logQ (-inf for non-candidates)
    logq_alpha float       correction strength (1.0 = full, 0 = off)
    """
    B = U.shape[0]

    # in-batch logits: column j = positive of anchor j
    s_in = (U @ V_pos.t()) / tau                          # [B, B]
    if logq_alpha != 0.0:
        s_in = s_in - logq_alpha * logq[pos_idx].unsqueeze(0)  # logQ per column (item as positive)

    # false negatives: i != j but pos[i] == pos[j]
    same = pos_idx.unsqueeze(0) == pos_idx.unsqueeze(1)  # [B, B]
    eye = torch.eye(B, dtype=torch.bool, device=U.device)
    s_in = s_in.masked_fill(same & ~eye, NEG_INF)

    # hard-neg logits: no logQ, + log(beta), padding -> -inf
    s_hn = (U.unsqueeze(1) * V_hn).sum(-1) / tau         # [B, m]
    if beta != 1.0:
        s_hn = s_hn + torch.log(torch.tensor(beta, device=U.device, dtype=s_hn.dtype))
    s_hn = s_hn.masked_fill(~hardneg_mask, NEG_INF)

    logits = torch.cat([s_in, s_hn], dim=1)              # [B, B + m]
    target = torch.arange(B, device=U.device)           # positive = diagonal
    return F.cross_entropy(logits, target)

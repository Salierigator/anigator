"""Blend of the retriever cosine and the ranker prediction inside each candidate group.

Pure numpy (no torch). Evaluation and serving score with these same functions.
"""
from __future__ import annotations

import numpy as np


def group_offsets(qid: np.ndarray) -> np.ndarray:
    """Contiguous qid (0..G-1, ascending) -> offsets [G+1]. The assert catches a badly written pool."""
    change = np.r_[True, qid[1:] != qid[:-1]]
    starts = np.flatnonzero(change)
    assert (qid[starts] == np.arange(len(starts))).all(), "qid must be dense and contiguous 0..G-1"
    return np.r_[starts, len(qid)]


def rank_norm(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rank normalized to [0, 1] per group (larger value -> closer to 1). Ties are broken by
    position, which is enough for blend."""
    out = np.empty(len(x), dtype=np.float64)
    for s, e in zip(offsets[:-1], offsets[1:]):
        n = e - s
        r = np.empty(n, dtype=np.float64)
        r[np.argsort(x[s:e], kind="stable")] = np.arange(n)
        out[s:e] = r / max(n - 1, 1)
    return out


def blend(cos: np.ndarray, pred: np.ndarray, offsets: np.ndarray, alpha: float) -> np.ndarray:
    """score = (1 - alpha) * rank_norm(cos) + alpha * rank_norm(pred) per group. Serving applies
    exactly this formula."""
    if alpha == 0.0:
        return cos.astype(np.float64)
    return (1 - alpha) * rank_norm(cos, offsets) + alpha * rank_norm(pred, offsets)

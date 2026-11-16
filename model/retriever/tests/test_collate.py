"""Collate: per-anchor history sampling, anchor removal, pad/mask, dropout, hard negatives."""
import numpy as np
import torch

from model.retriever import data as data_mod


def _collate(users, L=3, dropout=0.0, m=2):
    torch.manual_seed(7)
    c = data_mod.Collate(users, hist_dropout=dropout, m_hardneg=m, train_hist_len=L)
    return c


def test_history_sample_and_anchor_removed(users):
    c = _collate(users, L=3)
    batch = c([(0, 3), (1, 6), (2, 5)])
    hist = batch["history_ids"].numpy()
    mask = batch["history_mask"].numpy()

    # u0: len 4 > L=3 -> sampled with replacement from {2,3,4,5}; anchor 3 is masked
    assert set(hist[0]) <= {2, 3, 4, 5}
    assert not (mask[0] & (hist[0] == 3)).any(), "anchor not removed from the history"

    # u1: len 2 <= L -> all of [6,7], padded by repeating the last, mask of the true length; anchor 6 masked
    assert hist[1].tolist() == [6, 7, 7]
    assert mask[1].tolist() == [False, True, False]

    # u2: empty history -> all-False mask (the model substitutes h_empty)
    assert not mask[2].any()


def test_history_scores_aligned(users):
    c = _collate(users, L=3)
    batch = c([(1, 99)])                       # anchor not in the history
    hist = batch["history_ids"].numpy()[0]
    sc = batch["history_scores"].numpy()[0]
    # u1: ids [6,7,7] -> scores [5,0,0] (gathered at the same positions)
    assert hist.tolist() == [6, 7, 7]
    assert sc.tolist() == [5, 0, 0]


def test_hist_dropout_full(users):
    c = _collate(users, L=3, dropout=1.0)
    batch = c([(0, 3), (1, 6)])
    assert not batch["history_mask"].numpy().any(), "dropout=1 must drop the whole history"


def test_hardneg_sample_and_mask(users):
    c = _collate(users, L=3, m=2)
    batch = c([(0, 3), (1, 6), (2, 5)])
    ids = batch["hardneg_ids"].numpy()
    mask = batch["hardneg_mask"].numpy()
    # u0: 2 real, distinct hard negatives from the pool {8,9}
    assert mask[0].tolist() == [True, True]
    assert set(ids[0]) == {8, 9}
    # u1: 1 real (id 2) + 1 padding with mask False
    assert mask[1].tolist() == [True, False]
    assert ids[1][0] == 2
    # u2: no hard negatives -> all-False mask (pure in-batch loss)
    assert mask[2].tolist() == [False, False]


def test_sample_within_bounds_many(users):
    """Sampled many times, every history id stays within the user's own slice."""
    c = _collate(users, L=4)
    for _ in range(20):
        batch = c([(0, 99), (1, 99), (2, 99)])
        hist = batch["history_ids"].numpy()
        mask = batch["history_mask"].numpy()
        assert set(hist[0][mask[0]]) <= {2, 3, 4, 5}
        assert set(hist[1][mask[1]]) <= {6, 7}
        assert not mask[2].any()

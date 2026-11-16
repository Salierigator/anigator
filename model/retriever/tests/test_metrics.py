"""Eval protocol: seen-mask (mask seen - query, never the query), pooled hit-rate,
candidate_mask (rank only H), eval_history_batch prefix/cap."""
import numpy as np
import pytest
import torch

from model.retriever import metrics


class _StubModel:
    """item_cache = eye(N); encode_users returns a fixed U per user, in batch order."""

    def __init__(self, n_items, U_by_user):
        self.item_cache = torch.eye(n_items)
        self._U = U_by_user                          # dict user -> vector [N]
        self._calls = []

    def eval(self):
        pass

    def train(self):
        pass

    def encode_users(self, ub):
        # batch order recovered from gender_id, which the stub sets to user_idx
        users = ub["gender_id"].tolist()
        return torch.stack([self._U[u] for u in users])


def _stub_users(num_users, n_items):
    from model.retriever import data as data_mod
    ut = data_mod.UserTable.__new__(data_mod.UserTable)
    ut.num_users = num_users
    ut.hist_vals = np.zeros(1, np.int32)
    ut.hist_offs = np.zeros(num_users + 1, np.int64)     # empty histories (unused by the stub)
    ut.hist_scores = np.zeros(1, np.int8)
    ut.gender_id = np.arange(num_users, dtype=np.int64)  # user_idx planted in gender for the stub
    ut.joined_bucket = np.zeros(num_users, np.int64)
    return ut


def _logq(n):
    lq = torch.zeros(n)
    lq[0] = lq[1] = float("-inf")                        # PAD/OOV non-candidate
    return lq


def test_seen_masked_query_not_masked():
    N = 6
    # user 0 prefers item2 > item3 > item4 > item5; seen = {2,3}; query = {3}
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users(1, N)
    queries = {0: [3]}
    mask_ids = metrics.build_masks({0: np.array([2, 3])}, queries)
    assert mask_ids[0].tolist() == [2], "mask must be seen - query"
    out = metrics.evaluate(model, users, queries, _logq(N), [1, 2], mask_ids)
    # item2 (seen, not a query) is masked -> top1 = item3 = the query -> recall@1 = 1
    assert out["recall@1"] == 1.0, "the query was masked, or seen was not"


def test_without_mask_seen_pollutes():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users(1, N)
    out = metrics.evaluate(model, users, {0: [3]}, _logq(N), [1], {0: np.empty(0, np.int64)})
    assert out["recall@1"] == 0.0, "without the mask, item2 (seen) must take top1"


def test_pooled_hitrate_and_counts():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.1, 0.2]),
         1: torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 0.8])}
    model = _StubModel(N, U)
    users = _stub_users(2, N)
    queries = {0: [2, 3], 1: [4, 5]}
    masks = {0: np.empty(0, np.int64), 1: np.empty(0, np.int64)}
    out = metrics.evaluate(model, users, queries, _logq(N), [2], masks, pooled=True)
    assert out["n_users"] == 2 and out["n_pairs"] == 4
    assert out["recall@2"] == 1.0 and out["hitrate@2"] == 1.0


def test_candidate_mask_h_only():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users(1, N)
    cand = torch.zeros(N, dtype=torch.bool)
    cand[4] = cand[5] = True                             # "H only": rank items 4, 5 only
    out = metrics.evaluate(model, users, {0: [4]}, _logq(N), [1], {0: np.empty(0, np.int64)},
                           candidate_mask=cand)
    assert out["recall@1"] == 1.0, "items outside the candidate set must be excluded (2, 3 never ranked)"


def _stub_users_support(supp_scores: dict):
    """Stub UserTable with per-user support scores (for support_mean / the liked metric)."""
    from model.retriever import data as data_mod
    num_users = len(supp_scores)
    vals, scores, offs = [], [], [0]
    for u in range(num_users):
        sc = supp_scores[u]
        vals.extend([2] * len(sc))                       # any support anime_idx (>= 2)
        scores.extend(sc)
        offs.append(offs[-1] + len(sc))
    ut = data_mod.UserTable.__new__(data_mod.UserTable)
    ut.num_users = num_users
    ut.hist_vals = np.asarray(vals or [0], np.int32)
    ut.hist_offs = np.asarray(offs, np.int64)
    ut.hist_scores = np.asarray(scores or [0], np.int8)
    ut.gender_id = np.arange(num_users, dtype=np.int64)
    ut.joined_bucket = np.zeros(num_users, np.int64)
    return ut


def test_liked_recall_ndcg_hand_computed():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}  # ranking of real items desc: 2>3>4>5
    model = _StubModel(N, U)
    users = _stub_users_support({0: [6, 4]})            # rated u_mean = 5.0
    queries = {0: [2, 3, 4, 5]}
    query_scores = {0: [10, 5, 7, 0]}                      # liked = score>=1 & score>5 -> {2,4}
    masks = {0: np.empty(0, np.int64)}                     # no mask -> top = [2,3,4,5]
    out = metrics.evaluate(model, users, queries, _logq(N), [1, 2, 3], masks,
                           query_scores=query_scores)
    # R_liked=2 ({2,4}); item3 (score 5, not above the mean) and item5 (score 0) are out
    assert out["n_users_liked"] == 1
    assert out["liked_recall@1"] == pytest.approx(0.5)    # top1={2} -> 1/2
    assert out["liked_recall@2"] == pytest.approx(0.5)    # top2={2,3}, liked hit {2} -> 1/2
    assert out["liked_recall@3"] == pytest.approx(1.0)    # top3={2,3,4} -> {2,4} -> 2/2
    idcg2 = 1 + 1 / np.log2(3)                            # min(R_liked,2)=2 liked items first
    assert out["liked_ndcg@2"] == pytest.approx(1.0 / idcg2)  # only item2 (rank 0) is liked
    # the binary metrics are unchanged (same ranking and mask)
    assert out["recall@3"] == pytest.approx(3 / 4)


def test_liked_skips_user_with_no_liked():
    N = 6
    U = {0: torch.tensor([0.0, 0.0, 0.9, 0.8, 0.7, 0.6])}
    model = _StubModel(N, U)
    users = _stub_users_support({0: [9, 9]})           # u_mean = 9 -> no query score above 9
    queries = {0: [2, 3]}
    query_scores = {0: [8, 7]}
    out = metrics.evaluate(model, users, queries, _logq(N), [1, 2], {0: np.empty(0, np.int64)},
                           query_scores=query_scores)
    assert out["n_users"] == 1 and out["n_users_liked"] == 0
    assert out["liked_recall@2"] == 0.0                   # n_liked=0 -> defaults to 0


def test_eval_history_batch_prefix_cap(users):
    ids, mask, sc = users.eval_history_batch(np.array([0, 1, 2]), cap=2)
    assert ids.shape[1] == 2
    assert ids[0].tolist() == [2, 3], "prefix = top-by-score (lists are sorted desc)"
    assert mask[0].tolist() == [True, True]
    assert ids[1].tolist() == [6, 7] and sc[1].tolist() == [5, 0]
    assert not mask[2].any() and ids[2].tolist() == [0, 0], "empty row: pad 0 + mask False"

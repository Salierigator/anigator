"""metrics.py and blend.py invariants: recall/ndcg pinned to hand-computed values, rank_norm/blend, group_offsets."""
import numpy as np
import pytest

from model.twotower.ranker.core.blend import blend, group_offsets, rank_norm
from model.twotower.ranker.core.metrics import eval_pool


def test_group_offsets_dense():
    qid = np.array([0, 0, 0, 1, 1, 2])
    assert group_offsets(qid).tolist() == [0, 3, 5, 6]
    with pytest.raises(AssertionError):
        group_offsets(np.array([0, 0, 2, 2]))          # qid 1 missing


def test_recall_ndcg_hand_computed():
    # 1 group, 3 candidates with scores desc [3,2,1], labels [0,1,0], R_total=2 (1 query outside the pool)
    scores = np.array([3.0, 2.0, 1.0])
    labels = np.array([0, 1, 0], dtype=np.int8)
    off = np.array([0, 3])
    m = eval_pool(scores, labels, off, np.array([2]), ks=[1, 2, 3])
    assert m["recall@1"] == 0.0
    assert m["recall@2"] == pytest.approx(0.5)         # 1 hit / R=2
    assert m["recall@3"] == pytest.approx(0.5)         # the 2nd query is not in the pool
    dcg2 = 1 / np.log2(3)                              # hit at rank 2
    idcg2 = 1 + 1 / np.log2(3)                         # min(R,k)=2 relevant items on top
    assert m["ndcg@2"] == pytest.approx(dcg2 / idcg2)
    assert m["n_users"] == 1


def test_idcg_truncation_min_R_k():
    # R=5 > k=2 -> idcg sums only the first 2 discounts
    scores = np.array([2.0, 1.0])
    labels = np.array([1, 1], dtype=np.int8)
    m = eval_pool(scores, labels, np.array([0, 2]), np.array([5]), ks=[2])
    idcg2 = 1 + 1 / np.log2(3)
    assert m["ndcg@2"] == pytest.approx((1 + 1 / np.log2(3)) / idcg2) == pytest.approx(1.0)
    assert m["recall@2"] == pytest.approx(2 / 5)


def test_liked_recall_ndcg_hand_computed():
    # 1 group, 4 candidates, scores desc -> order [0,1,2,3]; liked = cand0, cand2; r_liked=3 (1 outside the pool)
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 1, 1, 0], dtype=np.int8)
    label_liked = np.array([1, 0, 1, 0], dtype=np.int8)
    m = eval_pool(scores, labels, np.array([0, 4]), np.array([3]), ks=[1, 2, 3],
                  label_liked=label_liked, r_liked=np.array([3]))
    assert m["n_users_liked"] == 1
    assert m["liked_recall@1"] == pytest.approx(1 / 3)    # top1 = cand0 (liked), denominator r_liked=3
    assert m["liked_recall@2"] == pytest.approx(1 / 3)    # top2 = {0,1}, liked hit {0}
    assert m["liked_recall@3"] == pytest.approx(2 / 3)    # top3 = {0,1,2}, liked {0,2}
    idcg2 = 1 + 1 / np.log2(3)                            # min(r_liked,2)=2 liked items on top
    assert m["liked_ndcg@2"] == pytest.approx(1.0 / idcg2)  # only cand0 (rank 0) is liked
    # binary metrics unchanged (same ranking)
    assert m["recall@3"] == pytest.approx(1.0)            # 3 labels / r_total=3


def test_liked_skips_user_with_no_liked():
    m = eval_pool(np.array([2.0, 1.0]), np.array([1, 1], dtype=np.int8), np.array([0, 2]),
                  np.array([2]), ks=[2],
                  label_liked=np.array([0, 0], dtype=np.int8), r_liked=np.array([0]))
    assert m["n_users_liked"] == 0
    assert m["liked_recall@2"] == 0.0                     # n_liked=0 -> defaults to 0


def test_pooled_hitrate():
    scores = np.array([2.0, 1.0, 2.0, 1.0])
    labels = np.array([1, 0, 0, 0], dtype=np.int8)
    off = np.array([0, 2, 4])
    m = eval_pool(scores, labels, off, np.array([1, 1]), ks=[1], pooled=True)
    assert m["hitrate@1"] == pytest.approx(0.5)        # 1 hit / 2 pairs
    assert m["n_pairs"] == 2


def test_rank_norm_range_and_monotonic():
    off = np.array([0, 4])
    x = np.array([0.1, 3.0, -2.0, 0.5])
    r = rank_norm(x, off)
    assert r.min() == 0.0 and r.max() == 1.0
    assert (np.argsort(r) == np.argsort(x)).all()      # order preserved


def test_blend_alpha0_is_cosine_alpha1_is_pred():
    off = np.array([0, 5])
    cos = np.random.default_rng(0).random(5)
    pred = np.random.default_rng(1).random(5)
    assert (np.argsort(blend(cos, pred, off, 0.0)) == np.argsort(cos)).all()
    assert (np.argsort(blend(cos, pred, off, 1.0)) == np.argsort(pred)).all()

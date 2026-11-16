"""features.py invariants on the real bundle: FEATURE_NAMES order, cold policy, determinism."""
import numpy as np
import pytest

from model.twotower.ranker import config
from model.twotower.ranker.core.features import CAT_COLS, FEATURE_NAMES, N_CROSS, ItemFeatures, build_frame

pytestmark = pytest.mark.skipif(
    not (config.EXPORT / "item_index.parquet").exists(), reason="needs export/twotower/")


@pytest.fixture(scope="module")
def itemfeat():
    return ItemFeatures.load(config.EXPORT)


def test_build_frame_column_order(itemfeat):
    cand = np.array([2, 3, 4])
    cross = {n: np.zeros(3, np.float32) for n in FEATURE_NAMES[:N_CROSS]}
    df = build_frame(itemfeat, cand, cross)
    assert list(df.columns) == FEATURE_NAMES


def test_cold_policy_no_leak(itemfeat):
    cold = itemfeat.is_cold
    assert cold.sum() > 0
    assert itemfeat.item["mal_score_missing"][cold].all(), "cold items must be flagged missing"
    assert (itemfeat.item["log_scored_by"][cold] == 0).all()
    assert (itemfeat.item["log_members"][cold] == 0).all()
    assert (itemfeat.item["log_favorites"][cold] == 0).all()
    assert itemfeat.item["rank_missing"][cold].all()
    # imputed to the warm median: every cold row gets the same value
    assert len(np.unique(itemfeat.item["mal_score"][cold])) == 1
    assert len(np.unique(itemfeat.item["popularity"][cold])) == 1


def test_categorical_codes_deterministic(itemfeat):
    again = ItemFeatures.load(config.EXPORT)
    for c in CAT_COLS:
        assert (itemfeat.item[c] == again.item[c]).all()
    assert (itemfeat.genres == again.genres).all()


def test_affinity_shapes(itemfeat):
    cand = np.array([[2, 3], [4, 5]])                  # batched [2, 2]
    g_pref = np.zeros((2, itemfeat.genres.shape[1]), np.float32)
    t_pref = np.zeros((2, itemfeat.themes.shape[1]), np.float32)
    ga, ta, go = itemfeat.affinity(cand, g_pref, t_pref)
    assert ga.shape == ta.shape == go.shape == (2, 2)
    assert (ga == 0).all()

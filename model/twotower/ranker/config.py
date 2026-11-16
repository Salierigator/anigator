"""Shared ranker constants: paths and build/train/eval knobs.

Paths come from `model.twotower.paths`; run every entry point from the repo root as
`python -m model.twotower.ranker.<module>`. The ranker reads only export/twotower/ (including details.csv and
profiles.csv), never the raw ratings or the retriever's training data: model/twotower/retriever/export/export.py
exports everything it needs.
"""
from __future__ import annotations

from model.twotower.paths import EXPORT, TRAIN

DATA = TRAIN / "ranker" / "data"
POOLS = DATA / "pools"                                 # eval pools (data_prep/build_eval.py)
DATASETS = DATA / "datasets"                           # train dataset (data_prep/build_train.py)
RUNS = TRAIN / "ranker" / "runs"                       # train_lgbm runs, leaderboard.csv, eval_selection.json

SEED = 42
NUM_THREADS = 4           # LightGBM threads for local training (kept low to limit heat)
K_POOL = 200              # rerank pool at serve time; train candidates = top-K_POOL
POOL_DEPTH = 500          # eval pools are stored this deep, so K=200 and K=500 score without re-encoding
TARGET_FRAC = 0.2         # share of a train user's positives held out as targets (mirrors EVAL_QUERY_FRAC)
N_TRAIN_USERS = 100_000   # train users sampled for the dataset
MIN_POS_TRAIN = 5         # min positives for a sampled train user
HIST_FEAT_CAP = 256       # history prefix (top by score) used for features (sims/affinity);
                          # encoding U always uses eval_history_cap from user_tower.pt
HIST_TOP64 = 64           # history prefix stored in the pool side tables (LightGBM does not read it)
CHUNK = 512               # users per encode/score/feature chunk

KS = [10, 50, 100, 200]   # metric K for selection (500 only in the pool-depth ablation)
ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]   # blend sweep (0 = cosine baseline)

GRADE_MAP = "10->4, 9->3, 7-8->2, 0/5/6->1, non-target->0"


def grade(score):
    """Relevance grade from the raw score (vectorized). Covers every positive, matching the
    retriever's definition (status in {completed, watching} and score not in [1, 4])."""
    import numpy as np

    s = np.asarray(score)
    return np.select([s >= 10, s >= 9, s >= 7], [4, 3, 2], default=1).astype(np.int8)

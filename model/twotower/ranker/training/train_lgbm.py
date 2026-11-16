"""Train LightGBM rankers: the sweep grid, its --relabel variants, or a --smoke check.

Train = datasets/train.parquet (graded labels 0-4, group = user, top-200 pool). Early stopping
uses pools/eval_val.parquet sliced to 200 (binary labels): LightGBM's internal ndcg only drives
early stopping, and the numbers that count are the two-stage metrics.py ones on eval_val,
computed after each run.

Writes train/twotower/ranker/runs/<run>/{model.txt,row.json} and train/twotower/ranker/runs/leaderboard.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (before lightgbm: two OpenMP runtimes segfault on macOS)
import lightgbm as lgb
import polars as pl

from model.twotower.ranker import config
from model.twotower.ranker.core.features import CAT_COLS, FEATURE_NAMES
from model.twotower.ranker.core.blend import blend
from model.twotower.ranker.core.metrics import eval_pool, sweep_best_alpha
from model.twotower.ranker.core.pools import load_pool_arrays

BASE = dict(
    objective="lambdarank",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    metric="ndcg",
    ndcg_eval_at=[10, 100],
    verbosity=-1,
    seed=42,
    num_threads=config.NUM_THREADS,
)
NUM_ROUNDS = 4000
EARLY_STOP = 100
ALPHAS = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
KS = [10, 50, 100, 200]


def _groups(qid: np.ndarray) -> np.ndarray:
    _, counts = np.unique(qid, return_counts=True)
    return counts


def relabel_label(tr: pl.DataFrame, mode: str) -> np.ndarray:
    """Train labels [N] for a grading mode.
    - 'default' = the prebuilt `label` column (config.grade: 10->4, 9->3, 7-8->2, 0/5/6->1).
    - 'steep'   = a sharper head (10->5, 9->4, 8->3, 7->2, else->1).
    - 'liked'   = the default grade + 1 if score > u_mean (ties the objective to the liked definition).
    steep/liked need the `target_score` column written by build_train."""
    if mode == "default":
        return tr["label"].to_numpy()
    if "target_score" not in tr.columns:
        raise SystemExit("relabel needs the target_score column: rebuild the dataset with build_train first.")
    ts = tr["target_score"].to_numpy()
    is_tgt = tr["label"].to_numpy() > 0
    if mode == "steep":
        lab = np.select([ts >= 10, ts >= 9, ts >= 8, ts >= 7], [5, 4, 3, 2], default=1)
    elif mode == "liked":
        um = tr["u_mean_score"].to_numpy()
        lab = config.grade(ts).astype(np.int16) + ((ts > um) & is_tgt)
    else:
        raise SystemExit(f"unknown relabel mode: {mode}")
    return (lab * is_tgt).astype(np.int8)


def load_datasets(data_dir: Path, max_groups: int | None = None, relabel: str = "default"):
    """(train Dataset, valid Dataset, valid arrays for the two-stage metrics). data_dir holds
    datasets/train.parquet + pools/eval_val{,_users}.parquet; relabel is the train grading mode."""
    tr = pl.read_parquet(data_dir / "datasets" / "train.parquet")
    if max_groups is not None:
        tr = tr.filter(pl.col("qid") < max_groups)
    dtrain = lgb.Dataset(
        tr.select(FEATURE_NAMES).to_pandas(), label=relabel_label(tr, relabel),
        group=_groups(tr["qid"].to_numpy()),
        feature_name=FEATURE_NAMES, categorical_feature=CAT_COLS, free_raw_data=True,
        params={"feature_pre_filter": False})   # lets the sweep go below the build-time min_data_in_leaf

    va_df, va_cos, va_lab, va_off, va_rt, va_users = load_pool_arrays(
        data_dir / "pools" / "eval_val.parquet",
        data_dir / "pools" / "eval_val_users.parquet", k=200, max_groups=max_groups)
    X_val = va_df.select(FEATURE_NAMES).to_pandas()
    va_liked = va_df["label_liked"].to_numpy().astype(np.int8)   # liked metrics (report only)
    va_rliked = va_users["r_liked"].to_numpy()
    dvalid = lgb.Dataset(X_val, label=va_lab, group=np.diff(va_off),
                         feature_name=FEATURE_NAMES, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    return dtrain, dvalid, (X_val, va_cos, va_lab, va_off, va_rt, va_liked, va_rliked)


def train_one(run_name: str, out_dir: Path, dtrain, dvalid, valid_arrays,
              relabel: str = "default", **overrides) -> dict:
    """Train one config and save model.txt + row.json (leaderboard row = two-stage val metrics at
    the best alpha). relabel only tags the row: main has already relabeled dtrain."""
    params = {**BASE, **overrides}
    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=NUM_ROUNDS, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=True), lgb.log_evaluation(200)])

    X_val, cos, lab, off, rt, lab_liked, r_liked = valid_arrays
    pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    best_a, best_m, _ = sweep_best_alpha(cos, pred, lab, off, rt, KS, ALPHAS)
    # liked metrics (report only) on the same ranking at the best alpha
    liked_m = eval_pool(blend(cos, pred, off, best_a), lab, off, rt, KS,
                        label_liked=lab_liked, r_liked=r_liked)

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "model.txt"), num_iteration=booster.best_iteration)
    row = {
        "run": run_name, "model_type": "lightgbm",
        "objective": params["objective"],
        "truncation": params.get("lambdarank_truncation_level"),
        "label_gain": params.get("label_gain"),
        "ndcg_eval_at": params.get("ndcg_eval_at"),
        "relabel": relabel,
        "learning_rate": params["learning_rate"], "num_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "best_iteration": booster.best_iteration,
        "best_alpha": best_a,
        **{f"val_{m}": round(best_m[m], 5)
           for m in ("recall@10", "recall@100", "ndcg@10", "ndcg@100")},
        **{f"val_{m}": round(liked_m[m], 5)
           for m in ("liked_recall@10", "liked_recall@100", "liked_ndcg@10")},
        "train_sec": round(time.time() - t0),
        "importance_gain": dict(zip(FEATURE_NAMES,
                                    booster.feature_importance("gain").round(1).tolist())),
    }
    (run_dir / "row.json").write_text(json.dumps(row, indent=2))
    print(f"[{run_name}] iter={row['best_iteration']} alpha={best_a} "
          f"ndcg@10={best_m['ndcg@10']:.4f} liked_ndcg@10={liked_m['liked_ndcg@10']:.4f} "
          f"liked_r@100={liked_m['liked_recall@100']:.4f} r@10={best_m['recall@10']:.4f} "
          f"({row['train_sec']}s)")
    return row


# LightGBM sweep grid: objective/truncation x label_gain x capacity x early-stopping metric.
# Every run uses the default graded labels; --relabel reruns RELABEL_SWEEP under another grading.
# Ordered by expected impact (objective/truncation/label_gain first), so the sweep can stop early.
SWEEP = [
    # objective: xendcg vs lambdarank truncation (lower = gradient focused on the head -> ndcg@10)
    ("xendcg",            dict(objective="rank_xendcg")),
    ("lrank_t10",         dict(lambdarank_truncation_level=10)),
    ("lrank_t20",         dict(lambdarank_truncation_level=20)),
    ("lrank_t30",         dict(lambdarank_truncation_level=30)),
    ("lrank_t50",         dict(lambdarank_truncation_level=50)),
    # label_gain: reshape grade -> gain (lambdarank only), boosting high grades -> head/liked
    ("lrank_t20_gainTop", dict(lambdarank_truncation_level=20, label_gain=[0, 1, 3, 7, 31])),
    ("lrank_t20_gainExp", dict(lambdarank_truncation_level=20, label_gain=[0, 3, 7, 15, 31])),
    ("lrank_t20_gainLin", dict(lambdarank_truncation_level=20, label_gain=[0, 1, 2, 3, 4])),
    # early stopping on the head (ndcg@10 only instead of [10, 100])
    ("xendcg_es10",       dict(objective="rank_xendcg", ndcg_eval_at=[10])),
    ("lrank_t10_es10",    dict(lambdarank_truncation_level=10, ndcg_eval_at=[10])),
    # capacity / lr (xendcg)
    ("xendcg_l127",       dict(objective="rank_xendcg", num_leaves=127)),
    ("xendcg_l255_mdl50", dict(objective="rank_xendcg", num_leaves=255, min_data_in_leaf=50)),
    ("xendcg_lr03",       dict(objective="rank_xendcg", learning_rate=0.03)),
    ("xendcg_lr10",       dict(objective="rank_xendcg", learning_rate=0.1)),
    # capacity (lambdarank t20) + feature/bagging
    ("lrank_t20_l127",    dict(lambdarank_truncation_level=20, num_leaves=127)),
    ("xendcg_ff09",       dict(objective="rank_xendcg", feature_fraction=0.9, bagging_fraction=0.9)),
]

# Configs rerun under each relabel mode (--relabel steep|liked).
RELABEL_SWEEP = [
    ("xendcg",    dict(objective="rank_xendcg")),
    ("lrank_t20", dict(lambdarank_truncation_level=20)),
]


def leaderboard(models_dir: Path) -> None:
    """Collect runs/*/row.json, print the table and write runs/leaderboard.csv (sorted by val_ndcg@10)."""
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(models_dir.glob("*/row.json"))]
    if not rows:
        return
    df = pd.DataFrame(rows).drop(columns=["importance_gain"], errors="ignore") \
        .sort_values("val_ndcg@10", ascending=False)
    df.to_csv(models_dir / "leaderboard.csv", index=False)
    print("\n=== leaderboard (val_ndcg@10) ===")
    print(df.to_string(index=False))


def main() -> None:
    """Run the sweep grid locally -> runs/<run>/{model.txt,row.json} + leaderboard.csv.
    --smoke: 3k groups and one config (end-to-end check, not real numbers)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--relabel", choices=["default", "steep", "liked"], default="default",
                    help="train label grading (steep/liked run RELABEL_SWEEP)")
    ap.add_argument("--subset", type=int, default=None,
                    help="coarse sweep on the first N train groups (qid < N); confirm the winner on the full set")
    args = ap.parse_args()
    max_groups = 3_000 if args.smoke else args.subset

    dtrain, dvalid, valid_arrays = load_datasets(config.DATA, max_groups=max_groups,
                                                 relabel=args.relabel)
    global NUM_ROUNDS
    if args.smoke:
        NUM_ROUNDS = 50
        sweep = [("smoke_xendcg", dict(objective="rank_xendcg"))]
    elif args.relabel != "default":
        sweep = RELABEL_SWEEP
    else:
        sweep = SWEEP

    for run_name, overrides in sweep:
        name = run_name if args.relabel == "default" else f"{run_name}_{args.relabel}"
        if (config.RUNS / name / "row.json").exists():   # resume: skip finished configs
            print(f"[{name}] skip: row.json exists (resume)")
            continue
        train_one(name, config.RUNS, dtrain, dvalid, valid_arrays,
                  relabel=args.relabel, **overrides)
    leaderboard(config.RUNS)


if __name__ == "__main__":
    main()

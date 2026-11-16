"""Named Two-Tower runs: train one, sweep a grid of them, rank them on the leaderboards.

A run is a folder <runs-dir>/<name>/ holding best.pt (written by train.fit), history.json,
row.json (warm val + test) and cold_row.json (val_cold, added by `leaderboard`). A run with
row.json is done and gets skipped, so relaunching an interrupted job resumes it. test_cold is the
final exam and is never scored here.

Subcommands: `train` (one run on BASE plus an optional preset), `sweep` (every config of a grid
in SWEEPS) and `leaderboard` (score the runs, write runs.csv and cold_runs.csv). Each takes
--runs-dir; the default is train/twotower/retriever/runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import time
from pathlib import Path

import pandas as pd

from model.twotower.retriever import config
from model.twotower.retriever.core import checkpoint, data, metrics
from model.twotower.retriever.training import search, train

# Every named run and sweep starts from this config. batch_size 16384 needs an A100; on a T4 pass
# --batch-size 8192.
BASE = dict(batch_size=16384, m_hardneg=5, eval_every_steps=500, num_workers=2)

# The served checkpoint was trained with BASE + "final". Early stopping on warm val recall@200 ends
# the run, so epochs is only a cap. min_delta stays 0: while a run still improves, its gain between
# evals 500 steps apart is often below 0.001, so a threshold that size stops it early.
PRESETS = {
    "final": dict(use_item_id=True, id_dim=128, id_dropout=0.15, history_source="embed",
                  train_hist_len=128, epochs=10, early_stop_patience=6, early_stop_min_delta=0.0,
                  eval_cold_in_loop=True),
}

# Sweeps rank configs cheaply: the final model's knobs, 15% of users, 2 epochs.
COARSE = dict(use_item_id=True, id_dim=128, id_dropout=0.15, history_source="embed",
              train_hist_len=128, epochs=2, train_user_frac=0.15)

SWEEPS = {
    # Do many in-batch negatives (large batches) drown out the hard negatives?
    "batch_hardneg": dict(batch_size=[512, 1024, 2048, 4096, 16384], beta=[0.5, 1.0, 2.0],
                          m_hardneg=[0, 5, 10, 20]),
}

# Machine-specific cfg fields, left out of leaderboard rows.
_ENV_FIELDS = ("device", "ckpt_dir", "train_data")
_WARM_COLS = ["run_name", "best_step", "val_recall@200", "test_recall@100", "test_recall@200",
              "test_ndcg@10", "test_liked_recall@200", "test_liked_ndcg@10"]
_COLD_COLS = ["run_name", "cold_n_users", "cold_recall@100", "cold_recall@200", "cold_ndcg@10",
              "cold_hit@200", "cold_honly_recall@200", "cold_liked_recall@200"]


def load_run(run_dir: Path, device: str | None = None):
    """Rebuild a run's best checkpoint from the cfg saved in it, on this machine's data and device.
    Returns (cfg, ckpt, model, users, logq) with the item cache warm."""
    ckpt = checkpoint.load(Path(run_dir) / "best.pt")
    cfg = checkpoint.saved_config(ckpt["cfg"])
    cfg.device = device or config.auto_device()
    cfg.train_data = config.TRAIN_DATA
    _, logq, _, users, model = train.build(cfg)
    model.load_state_dict(ckpt["model"])
    model.refresh_item_cache()
    return cfg, ckpt, model, users, logq


def warm_metrics(cfg, model, users, logq) -> dict:
    """Warm val and test (mask = seen - query) at the leaderboard cut-offs, with liked-metrics."""
    seen = data.load_eval_seen(cfg.train_data)
    out = {}
    for split in ("val", "test"):
        ds = data.ExamplesDataset(cfg.train_data, split)
        queries = metrics.group_examples(ds.user_idx, ds.anime_idx)
        m = metrics.evaluate(model, users, queries, logq, cfg.eval_ks,
                             metrics.build_masks(seen, queries),
                             eval_history_cap=cfg.eval_history_cap,
                             query_scores=metrics.group_scores(ds.user_idx, ds.score))
        for k in (100, 200):
            out[f"{split}_recall@{k}"] = round(float(m[f"recall@{k}"]), 4)
            out[f"{split}_liked_recall@{k}"] = round(float(m[f"liked_recall@{k}"]), 4)
        out[f"{split}_ndcg@10"] = round(float(m["ndcg@10"]), 4)
        out[f"{split}_liked_ndcg@10"] = round(float(m["liked_ndcg@10"]), 4)
        out[f"{split}_n_users_liked"] = int(m["n_users_liked"])
    return out


def cold_metrics(cfg, model, users, logq) -> dict:
    """val_cold with new items encoded id -> OOV: the full catalog, plus recall with only the new
    items as candidates. Leaves the item cache cold."""
    full = metrics.run_cold_eval(model, users, cfg.train_data, logq, cfg.eval_ks, split="val",
                                 eval_history_cap=cfg.eval_history_cap)
    honly = metrics.run_cold_eval(model, users, cfg.train_data, logq, cfg.eval_ks, split="val",
                                  eval_history_cap=cfg.eval_history_cap, h_only=True)
    out = {"cold_n_users": full["n_users"], "cold_n_pairs": full["n_pairs"]}
    for k in (100, 200):
        out[f"cold_recall@{k}"] = round(float(full[f"recall@{k}"]), 4)
        out[f"cold_liked_recall@{k}"] = round(float(full[f"liked_recall@{k}"]), 4)
        out[f"cold_hit@{k}"] = round(float(full[f"hitrate@{k}"]), 4)
        out[f"cold_honly_recall@{k}"] = round(float(honly[f"recall@{k}"]), 4)
    out["cold_ndcg@10"] = round(float(full["ndcg@10"]), 4)
    out["cold_liked_ndcg@10"] = round(float(full["liked_ndcg@10"]), 4)
    out["cold_n_users_liked"] = int(full["n_users_liked"])
    return out


def _row(name: str, cfg, ckpt: dict, scores: dict, **extra) -> dict:
    knobs = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _ENV_FIELDS}
    return {"run_name": name, **extra, "best_step": ckpt.get("step"), **knobs, **scores}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, default=float))


def run_experiment(name: str, runs_dir: Path = config.RUNS, **overrides) -> dict:
    """Train BASE + overrides into <runs_dir>/<name>/, then score its best checkpoint into row.json."""
    run_dir = Path(runs_dir) / name
    cfg = config.TwoTowerConfig(**{**BASE, **overrides, "ckpt_dir": run_dir})
    t0 = time.time()
    history = train.fit(cfg)[1]
    secs = round(time.time() - t0)
    _write_json(run_dir / "history.json", history)
    cfg, ckpt, model, users, logq = load_run(run_dir, cfg.device)
    row = _row(name, cfg, ckpt, warm_metrics(cfg, model, users, logq),
               time=time.strftime("%Y-%m-%d %H:%M"), secs=secs)
    _write_json(run_dir / "row.json", row)
    print(f"{name}: test recall@200 {row['test_recall@200']}")
    return row


def _score_run(run_dir: Path, force: bool, device: str | None) -> None:
    """Write the run's missing row.json / cold_row.json (both with force)."""
    warm_file, cold_file = run_dir / "row.json", run_dir / "cold_row.json"
    redo_warm, redo_cold = force or not warm_file.exists(), force or not cold_file.exists()
    if not (redo_warm or redo_cold):
        return
    cfg, ckpt, model, users, logq = load_run(run_dir, device)
    if redo_warm:  # before cold_metrics, which leaves the item cache cold
        _write_json(warm_file, _row(run_dir.name, cfg, ckpt, warm_metrics(cfg, model, users, logq)))
    if redo_cold:
        _write_json(cold_file, _row(run_dir.name, cfg, ckpt, cold_metrics(cfg, model, users, logq)))
    print(f"scored {run_dir.name}")


def leaderboard(runs_dir: Path = config.RUNS, force: bool = False, device: str | None = None):
    """Score every <runs_dir>/<run>/best.pt that lacks cached rows, then write runs.csv (warm,
    best test recall@200 first) and cold_runs.csv (val_cold, best cold recall@200 first)."""
    runs_dir = Path(runs_dir)
    run_dirs = sorted(p.parent for p in runs_dir.glob("*/best.pt"))
    if not run_dirs:
        print(f"no run with best.pt under {runs_dir}")
        return
    for run_dir in run_dirs:
        _score_run(run_dir, force, device)
    for file, key, cols in (("row.json", "test_recall@200", _WARM_COLS),
                            ("cold_row.json", "cold_recall@200", _COLD_COLS)):
        df = pd.DataFrame([json.loads((d / file).read_text()) for d in run_dirs])
        df = df.sort_values(key, ascending=False)
        out = runs_dir / ("runs.csv" if file == "row.json" else "cold_runs.csv")
        df.to_csv(out, index=False)
        print(f"\n{out}\n{df[cols].to_string(index=False)}")


def _done(runs_dir: Path, name: str) -> bool:
    return (runs_dir / name / "row.json").exists()


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs-dir", type=Path, default=config.RUNS,
                        help="folder holding one sub-folder per run (default: %(default)s)")
    ap = argparse.ArgumentParser(description="Train, sweep and rank named Two-Tower runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", parents=[common], help="train one run on BASE (+ a preset)")
    p.add_argument("name", help="run folder name")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--batch-size", type=int, help="override BASE's 16384, e.g. 8192 on a T4")
    p.add_argument("--device", help="default: cuda, else mps, else cpu")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset, 1 epoch, folder <name>_smoke: checks the pipeline only")

    p = sub.add_parser("sweep", parents=[common], help="train every config of a grid sweep")
    p.add_argument("name", choices=sorted(SWEEPS))
    p.add_argument("--dry-run", action="store_true", help="print run names without training")

    p = sub.add_parser("leaderboard", parents=[common],
                       help="score the runs, write runs.csv and cold_runs.csv")
    p.add_argument("--force", action="store_true", help="re-score runs that already have rows")
    p.add_argument("--device", help="default: cuda, else mps, else cpu")
    args = ap.parse_args()

    if args.cmd == "train":
        name = f"{args.name}_smoke" if args.smoke else args.name
        if _done(args.runs_dir, name):
            print(f"skip {name}: row.json exists")
            return
        overrides = dict(PRESETS.get(args.preset, {}))
        if args.smoke:
            overrides.update(train.SMOKE)
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if args.device is not None:
            overrides["device"] = args.device
        run_experiment(name, args.runs_dir, **overrides)
    elif args.cmd == "sweep":
        search.run_search(functools.partial(run_experiment, runs_dir=args.runs_dir),
                          search.iter_configs(SWEEPS[args.name], method="grid", fixed=COARSE),
                          exists_fn=functools.partial(_done, args.runs_dir), dry_run=args.dry_run)
    else:
        leaderboard(args.runs_dir, args.force, args.device)


if __name__ == "__main__":
    main()

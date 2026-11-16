"""Two-stage eval on the pools built by build_eval, and model selection.

Modes:
  --baseline-only   SANITY GATE: rank the pools by cos_uv; the metrics MUST reproduce
                    export/twotower/eval_reference.json (recall within 2e-3): same encoder/mask/queries
                    as the retriever. A failure means the harness drifted from the protocol: do
                    not train on it.
  (default)         Load the LightGBM models (*.txt) in train/twotower/ranker/runs/ -> sweep the blend
                    alpha on VAL -> Pareto-select against cosine -> report TEST + val_cold
                    -> write train/twotower/ranker/runs/eval_selection.json (read by export.py).
  --k 500           Pool-depth ablation (the default slices 200 from the 500-deep pools).
  --final-exam      test_cold, scored ONCE when the pipeline is final; needs the retriever's
                    `export --final-exam` first.

Selection (VAL only): Pareto >= cosine on {r@10, r@100, ndcg@10, ndcg@100} with ndcg@10 strictly
greater -> max ndcg@10; fallback: max ndcg@10 subject to r@100 >= cosine. (r@200 is the pool
ceiling at K=200, so it is left out.)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch  # noqa: F401  (before lightgbm: two OpenMP runtimes segfault on macOS)

from model.twotower.ranker import config
from model.twotower.ranker.core.features import FEATURE_NAMES
from model.twotower.ranker.core.blend import blend
from model.twotower.ranker.core.metrics import eval_pool, fmt

SEL_METRICS = ["recall@10", "recall@100", "ndcg@10", "ndcg@100"]
# Report only, never used for selection (selection stays on binary relevance).
LIKED_METRICS = ["liked_recall@100", "liked_recall@200", "liked_ndcg@10"]


def run_label(p: Path) -> str:
    """runs/<run_name>/model.txt -> run_name (a bare <name>.txt -> name)."""
    return p.parent.name if p.stem == "model" else p.stem


def liked_arrays(df, users):
    """label_liked (per candidate) + r_liked (per user) from a loaded pool; (None, None) if the pool
    lacks those columns."""
    if "label_liked" not in df.columns or "r_liked" not in users.columns:
        return None, None
    return df["label_liked"].to_numpy().astype(np.int8), users["r_liked"].to_numpy()


def load_pool(split: str, k: int):
    from model.twotower.ranker.core.pools import load_pool_arrays
    return load_pool_arrays(config.POOLS / f"eval_{split}.parquet",
                            config.POOLS / f"eval_{split}_users.parquet", k)


def predict(model_path: Path, df: pl.DataFrame) -> np.ndarray:
    """Score a pool with a LightGBM booster (.txt)."""
    if model_path.suffix == ".txt":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(model_path))
        X = df.select(FEATURE_NAMES).to_pandas()
        return booster.predict(X)
    raise SystemExit(f"unrecognized model: {model_path}")


def sweep(cos, pred, labels, offsets, r_total, ks, alphas, label_liked=None, r_liked=None):
    """{alpha: metrics} for one model; alpha = 0 is the cosine baseline."""
    return {a: eval_pool(blend(cos, pred, offsets, a), labels, offsets, r_total, ks,
                         pooled=False, label_liked=label_liked, r_liked=r_liked)
            for a in alphas}


def pareto_pick(results: dict[tuple, dict], base: dict) -> tuple | None:
    """results: {(model_name, alpha): metrics}. Pareto >= base on every SEL_METRICS with ndcg@10
    strictly greater, else fallback r@100 >= base; returns ((model_name, alpha), is_pareto) or None."""
    pareto = {k: v for k, v in results.items()
              if all(v[m] >= base[m] - 1e-9 for m in SEL_METRICS)
              and v["ndcg@10"] > base["ndcg@10"]}
    pool = pareto or {k: v for k, v in results.items()
                      if v["recall@100"] >= base["recall@100"] - 1e-9}
    if not pool:
        return None
    return max(pool, key=lambda k: pool[k]["ndcg@10"]), bool(pareto)


def baseline_gate() -> None:
    """The cosine baseline on the pools must reproduce export/twotower/eval_reference.json (recall within
    2e-3, same n_users)."""
    ref = json.loads((config.EXPORT / "eval_reference.json").read_text())
    ks_full = [10, 50, 100, 200, 500]
    ok = True
    for split, key in (("val", "val_warm"), ("test", "test_warm"), ("val_cold", "val_cold")):
        if not (config.POOLS / f"eval_{split}.parquet").exists():
            print(f"[skip] eval_{split} not built")
            continue
        df, cos, labels, offsets, r_total, users = load_pool(split, config.POOL_DEPTH)
        ll, rl = liked_arrays(df, users)
        m = eval_pool(cos, labels, offsets, r_total, ks_full, pooled=split.endswith("cold"),
                      label_liked=ll, r_liked=rl)
        print(f"[{split:8s}] {fmt(m, ks_full)}")
        if "n_users_liked" in m:
            print(f"           liked: " + "  ".join(f"{x}={m[x]:.4f}" for x in LIKED_METRICS)
                  + f"  (n_liked={m['n_users_liked']:,})")
        for k in ks_full:
            d = abs(m[f"recall@{k}"] - ref[key][f"recall@{k}"])
            if d > 2e-3:
                print(f"  FAIL recall@{k}: pool {m[f'recall@{k}']:.4f} vs ref "
                      f"{ref[key][f'recall@{k}']:.4f} (diff {d:.4f})")
                ok = False
        assert m["n_users"] == ref[key]["n_users"], \
            f"{split}: n_users {m['n_users']} != ref {ref[key]['n_users']}"
    if not ok:
        raise SystemExit("SANITY GATE FAIL: the pools drift from the retriever protocol; stop (do not train).")
    print("sanity gate PASS: cosine baseline reproduces eval_reference (encoder/mask/queries match)")


def report_split(split: str, k: int, models: list[Path], alphas, ks):
    df, cos, labels, offsets, r_total, users = load_pool(split, k)
    ll, rl = liked_arrays(df, users)
    base = eval_pool(cos, labels, offsets, r_total, ks, pooled=split.endswith("cold"),
                     label_liked=ll, r_liked=rl)
    cols = SEL_METRICS + (LIKED_METRICS if "n_users_liked" in base else [])
    print(f"\n=== {split} (K={k}, n={base['n_users']:,}"
          + (f", n_liked={base['n_users_liked']:,}" if "n_users_liked" in base else "") + ") ===")
    print(f"  {'config':<28}" + "".join(f"{m:>16}" for m in cols))
    print(f"  {'cosine(baseline)':<28}" + "".join(f"{base[m]:>16.4f}" for m in cols))
    results = {}
    for mp in models:
        pred = predict(mp, df)
        for a, m in sweep(cos, pred, labels, offsets, r_total, ks, alphas, ll, rl).items():
            if a == 0.0:
                continue
            results[(run_label(mp), a)] = m
            print(f"  {f'{run_label(mp)}|a{a}':<28}"
                  + "".join(f"{m[x]:>16.4f}" for x in cols))
    return base, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage eval + selection")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--k", type=int, default=config.K_POOL, choices=[200, 500])
    ap.add_argument("--models", nargs="*", type=Path,
                    help="default: every */model.txt and *.txt in train/twotower/ranker/runs/")
    ap.add_argument("--final-exam", action="store_true",
                    help="score test_cold ONCE (needs the retriever's export --final-exam and an existing selection)")
    args = ap.parse_args()
    t0 = time.time()

    baseline_gate()
    if args.baseline_only:
        return

    if args.final_exam:
        run_final_exam()
        return

    models = args.models or sorted(
        p for pat in ("*/model.txt", "*.txt")
        for p in config.RUNS.glob(pat))
    if not models:
        raise SystemExit(f"no model in {config.RUNS}: run train_lgbm first")
    print(f"models: {[run_label(m) for m in models]}")

    base_val, val_res = report_split("val", args.k, models, config.ALPHAS, config.KS)
    picked = pareto_pick(val_res, base_val)
    if picked is None:
        raise SystemExit("no config is >= cosine: check the models and features")
    (model_name, alpha), is_pareto = picked
    print(f"\n>>> PICKED (val): {model_name} alpha={alpha} "
          f"[{'Pareto-dominates cosine' if is_pareto else 'fallback r@100>=cosine'}]")

    chosen = next(m for m in models if run_label(m) == model_name)
    base_test, test_res = report_split("test", args.k, [chosen], [alpha], config.KS)
    test_m = test_res[(model_name, alpha)]
    sel_cols = SEL_METRICS + ([m for m in LIKED_METRICS if m in test_m])
    print("    test diff vs cosine: " + "  ".join(
        f"{m}={test_m[m]:.4f}({test_m[m] - base_test[m]:+.4f})" for m in sel_cols))

    base_cold, cold_res = report_split("val_cold", args.k, [chosen], [alpha], config.KS)
    cold_m = cold_res[(model_name, alpha)]

    sel = {
        "model": str(chosen), "model_name": model_name, "blend_alpha": alpha,
        "k_retrieve": args.k, "pareto": is_pareto,
        "val_metrics": val_res[(model_name, alpha)], "baseline_val": base_val,
        "test_metrics": test_m, "baseline_test": base_test,
        "val_cold_metrics": cold_m, "baseline_val_cold": base_cold,
        "pool_ceiling": {"val": base_val[f"recall@{args.k}"],
                         "test": base_test[f"recall@{args.k}"]},
    }
    config.RUNS.mkdir(exist_ok=True)
    (config.RUNS / "eval_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\nselection -> {config.RUNS / 'eval_selection.json'}  ({time.time() - t0:.0f}s)")


def run_final_exam() -> None:
    """FINAL EXAM: score test_cold exactly once, with the configuration fixed in eval_selection.json."""
    assert (config.EXPORT / "eval_queries_test_cold.parquet").exists(), \
        "run `venv/bin/python -m model.twotower.retriever.export.export --final-exam` first"
    sel = json.loads((config.RUNS / "eval_selection.json").read_text())
    if not (config.POOLS / "eval_test_cold.parquet").exists():
        from model.twotower.ranker.data_prep import build_eval
        from model.twotower.ranker.core.features import ItemFeatures
        from model.twotower.ranker.data_prep.bundle import UsersHistory, account_age_by_user, load_eval_seen
        from model.twotower.retriever.inference import load_user_encoder
        enc, meta = load_user_encoder("cpu")
        itemfeat = ItemFeatures.load(config.EXPORT)
        build_eval.build_split("test_cold", enc, meta.get("eval_history_cap", 1024), itemfeat,
                               enc.item_cache.numpy(), UsersHistory(), load_eval_seen(),
                               account_age_by_user())
    k, alpha = sel["k_retrieve"], sel["blend_alpha"]
    df, cos, labels, offsets, r_total, users = load_pool("test_cold", k)
    ll, rl = liked_arrays(df, users)
    base = eval_pool(cos, labels, offsets, r_total, config.KS, pooled=True,
                     label_liked=ll, r_liked=rl)
    pred = predict(Path(sel["model"]), df)
    m = eval_pool(blend(cos, pred, offsets, alpha), labels, offsets, r_total,
                  config.KS, pooled=True, label_liked=ll, r_liked=rl)
    print(f"\n=== FINAL EXAM test_cold (K={k}, alpha={alpha}, model={sel['model_name']}) ===")
    print(f"  cosine : {fmt(base, config.KS)}")
    print(f"  ranker : {fmt(m, config.KS)}")
    sel["test_cold_metrics"] = m
    sel["baseline_test_cold"] = base
    (config.RUNS / "eval_selection.json").write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()

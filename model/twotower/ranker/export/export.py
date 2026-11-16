"""Export the selected ranker to export/twotower/ranker.txt + export/twotower/ranker_meta.json
(only these two files).

Reads train/twotower/ranker/runs/eval_selection.json (written by eval.py after the blend sweep and Pareto
selection) and train/twotower/ranker/data/datasets/build_meta.json. The service serves LightGBM (.txt)
boosters only, so any other winner is refused.

There are no CLI options: running the module, even with --help, exports right away.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch  # noqa: F401  (before lightgbm)
import lightgbm as lgb

from model.twotower.ranker import config
from model.twotower.ranker.core.features import CAT_COLS, FEATURE_NAMES


def main() -> None:
    sel = json.loads((config.RUNS / "eval_selection.json").read_text())
    model_path = Path(sel["model"])
    if model_path.suffix != ".txt":
        raise SystemExit(
            f"winner {model_path.name} is not a LightGBM .txt model; the service only serves "
            "LightGBM boosters.")

    booster = lgb.Booster(model_file=str(model_path))
    assert booster.feature_name() == FEATURE_NAMES, "model feature order differs from FEATURE_NAMES"
    shutil.copyfile(model_path, config.EXPORT / "ranker.txt")

    bmeta = json.loads((config.DATASETS / "build_meta.json").read_text())
    meta = {
        "model_type": "lightgbm", "model_run": sel["model_name"],
        "best_iteration": booster.num_trees(),
        "feature_names": FEATURE_NAMES, "categorical_features": CAT_COLS,
        "grading": config.GRADE_MAP,
        "k_retrieve": sel["k_retrieve"], "blend_alpha": sel["blend_alpha"],
        "blend": "score = (1-alpha)*rank_norm(cos_uv) + alpha*rank_norm(ranker_pred)",
        "cold_feature_policy": "is_cold -> mal_score/scored_by/members/favorites/popularity/"
                               "rank impute-as-missing + flag (features.py)",
        "cold_serving": {
            "mode": "separate_channel_cosine",
            "rule": "main list = ranker rerank of the WARM candidates (is_cold filtered out of the "
                    "pool before reranking); cold items are served in their own section, ordered "
                    "by retriever cosine. Cold items never go through the ranker/blend.",
        },
        "hist_feat_cap": config.HIST_FEAT_CAP, "eval_history_cap": 1024,
        "pareto_vs_cosine": sel["pareto"],
        "val_metrics": sel["val_metrics"], "baseline_val": sel["baseline_val"],
        "test_metrics": sel["test_metrics"], "baseline_test": sel["baseline_test"],
        "val_cold_metrics": sel["val_cold_metrics"],
        "baseline_val_cold": sel["baseline_val_cold"],
        "pool_ceiling": sel["pool_ceiling"],
        "feature_importance_gain": dict(zip(
            FEATURE_NAMES, booster.feature_importance("gain").round(1).tolist())),
        "train_provenance": {k: bmeta[k] for k in
                             ("n_groups_kept", "rows", "k_pool", "pos_rate", "seed",
                              "git_rev", "source_checkpoint")},
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.EXPORT / "ranker_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"exported export/twotower/ranker.txt ({sel['model_name']}, alpha={sel['blend_alpha']}, "
          f"K={sel['k_retrieve']}) + ranker_meta.json")
    print(f"  val  ndcg@10 {sel['baseline_val']['ndcg@10']:.4f} -> "
          f"{sel['val_metrics']['ndcg@10']:.4f}")
    print(f"  test ndcg@10 {sel['baseline_test']['ndcg@10']:.4f} -> "
          f"{sel['test_metrics']['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()

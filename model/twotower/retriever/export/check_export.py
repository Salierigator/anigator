"""Check the export/twotower/ bundle: rebuild the user encoder from it (as the service does) and run the
full eval protocol (mask = seen - query) on warm val/test and cold val.

Faithful to the firewall: the encoder comes from retriever/inference.py, which reads only
export/twotower/ (item_vectors.npy + user_tower.pt); best.pt is never used to encode. It reuses
metrics.evaluate, so the numbers compare directly with training. Writes eval_reference.json to
export/twotower/ (reference numbers for the ranker's sanity gate).

Reading the output:
  - [val warm] should be close to the metrics stored in best.pt (a few thousandths off is
    normal: the bundle encodes cold rows with OOV, the serve path, while training's warm cache
    used real ids).
  - [val cold] should be close to run_cold_eval (e.g. cold_runs.csv from `runs.py leaderboard`):
    the cache is identical.
  - test_cold is not touched here (final exam, scored once when the model is final).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model.twotower import paths
from model.twotower.retriever import config
from model.twotower.retriever.core import checkpoint
from model.twotower.retriever.core.data import UserTable, load_cold_mask, load_feature_spec, load_logq
from model.twotower.retriever.core.metrics import evaluate, load_eval_protocol, load_query_scores
from model.twotower.retriever.inference import load_user_encoder


class BundleModel:
    """The model interface metrics.evaluate needs (.item_cache/.encode_users/.eval/.train), backed
    by the bundle's UserEncoder: the encoder the ranker and the service load."""

    def __init__(self, out: Path, device: str):
        self.enc, meta = load_user_encoder(device, out)
        self.item_cache = self.enc.item_cache
        self.eval_history_cap = meta.get("eval_history_cap", 1024)

    def eval(self):
        return self

    def train(self):
        return self

    def encode_users(self, batch):
        return self.enc.encode(batch["history_ids"], batch["history_mask"], batch.get("history_scores"),
                               batch["gender_id"], batch["joined_bucket"])


def validate_ranker_exports(out: Path) -> None:
    """Invariants of the eval-protocol files exported for the ranker (eval_queries_* / eval_seen
    / users_history). A failure means a broken export or a broken final-exam discipline."""
    import polars as pl

    assert not (out / "eval_queries_test_cold.parquet").exists(), (
        "eval_queries_test_cold.parquet EXISTS: the final exam is scored once, when the pipeline "
        "is final; re-run `python -m model.twotower.retriever.export.export` (without --final-exam) to delete it.")

    uh = pl.read_parquet(out / "users_history.parquet")
    usplit = pl.read_parquet(out / "user_split.parquet")
    assert uh.height == usplit.height and set(uh["user_idx"]) == set(usplit["user_idx"]), \
        "users_history must cover exactly the users of user_split"

    seen_t = pl.read_parquet(out / "eval_seen.parquet")
    seen = {int(u): set(s) for u, s in zip(seen_t["user_idx"], seen_t["seen_ids"].to_list())}
    item_idx = pl.read_parquet(out / "item_index.parquet")
    is_cold = item_idx["is_cold"].to_numpy()

    for split in ("val", "test", "val_cold"):
        q = pl.read_parquet(out / f"eval_queries_{split}.parquet")
        src = pl.read_parquet(config.TRAIN_DATA / "examples" / f"split={split}" / "part-0.parquet")
        assert q.height == src.height, f"eval_queries_{split} rows {q.height} != source {src.height}"
        qa = q["anime_idx"].to_numpy()
        if split.endswith("_cold"):
            assert is_cold[qa].all(), f"eval_queries_{split}: every query must be is_cold=True"
        else:
            assert not is_cold[qa].any(), f"eval_queries_{split}: warm queries must not be is_cold"
        hist = {int(u): set(h) for u, h in
                uh.filter(pl.col("user_idx").is_in(q["user_idx"].unique().to_list()))
                  .select("user_idx", "history_ids").iter_rows()}
        for u, grp in q.group_by("user_idx"):
            uu, qs = int(u[0]), set(grp["anime_idx"].to_list())
            assert not (qs & hist.get(uu, set())), f"{split}: queries overlap the history (user {uu})"
            assert qs <= seen.get(uu, set()), f"{split}: queries not within seen (user {uu})"
        print(f"[ok  ] eval_queries_{split}: {q.height:,} rows, no query in history, queries within seen, "
              f"is_cold {'all' if split.endswith('_cold') else 'none'}")

    # history sorted by score desc: checked on a sample
    sample = uh.head(2000)
    for sc in sample["history_scores"].to_list():
        if sc:
            a = np.asarray(sc)
            assert (a[:-1] >= a[1:]).all(), "users_history: history_scores must be sorted desc"
    print(f"[ok  ] users_history: {uh.height:,} users, history sort score desc (sample 2k)")


def fmt(m: dict, ks) -> str:
    cols = " ".join(f"r@{k}={m[f'recall@{k}']:.4f}" for k in ks)
    nd = " ".join(f"ndcg@{k}={m[f'ndcg@{k}']:.4f}" for k in (10, 200))
    return f"{cols}  {nd}  (n_users={m['n_users']:,})"


def main():
    ap = argparse.ArgumentParser(description="Evaluate the exported bundle with the full eval protocol")
    ap.add_argument("--out", type=Path, default=paths.EXPORT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--final-exam", action="store_true",
                    help="score the retriever's test_cold ONCE: full catalog + honly (only H as "
                         "candidates) + liked, serve path -> export/twotower/test_cold_final_exam.json "
                         "(run after a normal export)")
    args = ap.parse_args()

    validate_ranker_exports(args.out)

    ks = [10, 50, 100, 200, 500]
    spec = load_feature_spec(config.TRAIN_DATA)
    users = UserTable(config.TRAIN_DATA, spec["hard_neg_cap"])
    logq = load_logq(config.TRAIN_DATA).to(args.device)
    enc = BundleModel(args.out, args.device)
    cap = enc.eval_history_cap

    print(f"Eval on the bundle ({args.out}): encoder from user_tower.pt + item_vectors.npy "
          f"(history cap {cap})\n")
    reference = {}
    for split in ("val", "test"):
        q_warm, m_warm, q_cold, m_cold = load_eval_protocol(config.TRAIN_DATA, split)
        m = evaluate(enc, users, q_warm, logq, ks, m_warm, eval_history_cap=cap)
        print(f"[{split:4s} warm] {fmt(m, ks)}")
        reference[f"{split}_warm"] = m
        if split == "val":  # cold is scored on val only: test_cold is the final exam
            mc = evaluate(enc, users, q_cold, logq, ks, m_cold, eval_history_cap=cap, pooled=True)
            print(f"[val  cold] {fmt(mc, ks)}")
            print("            hitrate " + " ".join(f"@{k}={mc[f'hitrate@{k}']:.4f}" for k in ks)
                  + f"  (n_pairs={mc['n_pairs']:,})")
            reference["val_cold"] = mc

    # Reference numbers for the ranker's sanity gate (its cosine two-stage baseline must reproduce
    # them to ~1e-3). Measured through the bundle (cold rows = OOV), so slightly off the checkpoint
    # metrics in CONTRACT.
    import json
    (args.out / "eval_reference.json").write_text(json.dumps(reference, indent=2))
    print(f"\n[ok  ] eval_reference.json written to {args.out} (reference for the ranker sanity gate)")

    if args.final_exam:
        # Retriever cold FINAL EXAM, scored once: test_cold on the full catalog + honly (only H as
        # candidates, like the UI's new-anime section, which ranks new anime among themselves)
        # + liked. Serve path (cold rows = OOV), like every other official number.
        _, _, q_tc, m_tc = load_eval_protocol(config.TRAIN_DATA, "test")  # cold queries of test = test_cold
        qs_tc = load_query_scores(config.TRAIN_DATA, "test_cold")
        cold_mask = load_cold_mask(config.TRAIN_DATA, enc.item_cache.shape[0])
        full = evaluate(enc, users, q_tc, logq, ks, m_tc, eval_history_cap=cap,
                        pooled=True, query_scores=qs_tc)
        honly = evaluate(enc, users, q_tc, logq, ks, m_tc, eval_history_cap=cap,
                         pooled=True, query_scores=qs_tc, candidate_mask=cold_mask)
        def _liked(m):
            return (" ".join(f"lr@{k}={m[f'liked_recall@{k}']:.4f}" for k in (100, 200))
                    + f" lndcg@10={m['liked_ndcg@10']:.4f} (n_liked={m['n_users_liked']:,})")
        print(f"\n=== FINAL EXAM test_cold retriever (serve-path, n_users={full['n_users']:,}) ===")
        print(f"  full-catalog : {fmt(full, ks)}")
        print(f"     + liked   : {_liked(full)}")
        print(f"  honly (H only): {fmt(honly, ks)}")
        print(f"     + liked   : {_liked(honly)}")
        (args.out / "test_cold_final_exam.json").write_text(json.dumps(
            {"note": "retriever serve path, scored once; full = ranking over the whole 22.8k "
                     "catalog, honly = candidates limited to H (the UI's new-anime section)",
             "full_catalog": full, "honly": honly}, indent=2))
        print(f"[ok  ] test_cold_final_exam.json written to {args.out}")

    ckpt_path = config.RUNS / "best.pt"
    if ckpt_path.exists():
        ck = checkpoint.load(ckpt_path)
        ref = ck.get("metrics", {})
        if ref:
            cols = " ".join(f"r@{k}={ref.get(f'recall@{k}', float('nan')):.4f}" for k in ks)
            print(f"\n[ref ] best.pt recorded (val warm, epoch={ck.get('epoch')}): {cols}")
            print("       -> [val warm] through the bundle should be close to this line (slightly off"
                  " since cold rows = OOV); [val cold] close to run_cold_eval.")


if __name__ == "__main__":
    main()

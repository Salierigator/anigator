"""Export the Two-Tower bundle from a checkpoint (best.pt) to export/.

Re-run whenever best.pt changes. Nothing is hard-coded: hyperparameters come from the cfg
pickled in the checkpoint, paths from model/retriever/config.py.

Writes:
    item_vectors.npy     [num_items, d] float32, L2-normalized, row == anime_idx (PAD/OOV included).
                         Cold items (H) are encoded with id -> OOV: their ids were never trained,
                         so the real id is noise, and OOV matches the serve path measured at cold eval.
    item_index.parquet   anime_idx -> mal_id (-1 for PAD/OOV) + is_cold
    user_tower.pt        user-side state_dict + pooling cfg + user_features spec
    user_split.parquet   username, user_idx, split
    eval_queries_{val,test,val_cold}.parquet + eval_seen.parquet + users_history.parquet
                         eval protocol and histories for the ranker (test_cold only with --final-exam)
    CONTRACT.md          shapes/dtypes + the checkpoint's metrics
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from model import paths
from model.retriever import checkpoint, config
from model.retriever.data import ItemTable, load_cold_mask, load_feature_spec
from model.retriever.model import ItemTower, TwoTower


def reconcile_spec_to_ckpt(spec: dict, sd: dict) -> list[tuple[str, int, int]]:
    """The local spec may have been regenerated after best.pt was trained (e.g. joined vocab
    6 -> 5). The checkpoint is the source of truth for the architecture, so the spec's vocab
    sizes are forced to the trained embedding shapes and load_state_dict cannot break.
    Returns the mismatches so the caller warns instead of hiding them."""
    targets = [(spec["item_features"][n], f"item_tower.cat_emb.cat_{n}.weight", f"item.{n}")
               for n, _ in ItemTower.CAT_FEATS]
    targets += [
        (spec["item_features"]["studios"], "item_tower.studio_emb.weight", "item.studios"),
        (spec["user_features"]["gender"], "user_tower.gender_emb.weight", "user.gender"),
        (spec["user_features"]["joined"], "user_tower.joined_emb.weight", "user.joined"),
    ]
    fixes = []
    for entry, key, label in targets:
        ck_vocab = sd[key].shape[0]
        if entry["vocab"] != ck_vocab:
            fixes.append((label, entry["vocab"], ck_vocab))
            entry["vocab"] = ck_vocab
    return fixes


def build_model(ckpt: dict, device: str) -> tuple[TwoTower, dict]:
    """Build TwoTower from the pickled cfg with local paths and device, then load the weights."""
    cfg = checkpoint.saved_config(ckpt["cfg"])
    cfg.train_data = config.TRAIN_DATA      # a cfg pickled on Colab carries Colab paths
    cfg.device = device
    spec = load_feature_spec(cfg.train_data)
    fixes = reconcile_spec_to_ckpt(spec, ckpt["model"])
    if fixes:
        print("WARNING: local spec differs from the checkpoint (train data regenerated after training?):")
        for label, spec_v, ck_v in fixes:
            print(f"  - {label}: spec vocab={spec_v} -> using checkpoint vocab={ck_v}")
        print("  Item vectors are unaffected, only the user side. Consider retraining or "
              "regenerating the train data.")
    item_table = ItemTable(cfg.train_data, cfg.synopsis_emb_file,
                           cfg.synopsis_low_info_file, cfg.use_synopsis).to(device)
    model = TwoTower(spec, cfg, item_table).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, spec


def export_item_vectors(model: TwoTower, out: Path) -> np.ndarray:
    """Serve-path cache: cold items (H) encoded with id -> OOV (real content), warm items with real ids."""
    cold_mask = load_cold_mask(config.TRAIN_DATA, model.item_table.num_items)
    model.refresh_item_cache(chunk=8192, cold_mask=cold_mask)
    vecs = model.item_cache.detach().cpu().numpy().astype(np.float32)
    np.save(out / "item_vectors.npy", vecs)
    return vecs


def export_item_index(num_items: int, out: Path) -> None:
    """anime_id_map holds only real anime (idx 2..N+1); build a dense mal_id of length
    num_items with PAD/OOV = -1."""
    amap = pl.read_parquet(config.TRAIN_DATA / "anime_id_map.parquet")
    mal_id = np.full(num_items, -1, dtype=np.int64)
    idx = amap["anime_idx"].to_numpy()
    mal_id[idx] = amap["mal_id"].to_numpy()
    is_cold = np.zeros(num_items, dtype=bool)
    is_cold[pl.read_parquet(config.TRAIN_DATA / "cold_items.parquet")["anime_idx"].to_numpy()] = True
    pl.DataFrame({
        "anime_idx": np.arange(num_items, dtype=np.int32),
        "mal_id": mal_id,
        "is_cold": is_cold,
    }).write_parquet(out / "item_index.parquet")


def export_user_tower(ckpt: dict, spec: dict, out: Path) -> None:
    """Keep the user-side state_dict (drop item_tower.*) and pack what the service needs to
    build the encoder without ItemTable, pooling params (score_weight/attn_*) included."""
    cfg = ckpt["cfg"]
    user_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("item_tower.")}
    if cfg.history_source == "embed":
        # embed: the history table is hist_emb.* inside user_sd; the service pools through it
        assert any(k.startswith("hist_emb.") for k in user_sd), "history_source=embed but hist_emb is missing"
    torch.save({
        "state_dict": user_sd,
        "d": cfg.d,
        "mlp_hidden": cfg.mlp_hidden,        # the service rebuilds UserTower(spec, d, hidden)
        "history_pool": cfg.history_pool,
        "score_pool": cfg.score_pool,
        "history_source": cfg.history_source,  # cache: pool over item_vectors.npy | embed: over hist_emb
        "eval_history_cap": cfg.eval_history_cap,
        "user_features": spec["user_features"],
        "special_idx": spec["special_idx"],
    }, out / "user_tower.pt")
    return user_sd


def export_eval_protocol(out: Path, final_exam: bool) -> dict[str, int]:
    """Eval protocol for the ranker, whose two-stage eval must score exactly the retriever's
    queries/seen/history: eval_queries_{split}, eval_seen (copied) and users_history (every
    user). test_cold (final exam) is exported only with --final-exam; otherwise an old copy is
    deleted so the ranker harness cannot score it by mistake."""
    counts = {}
    for split in ("val", "test", "val_cold") + (("test_cold",) if final_exam else ()):
        q = pl.read_parquet(config.TRAIN_DATA / "examples" / f"split={split}" / "part-0.parquet")
        q.select(pl.col("user_idx").cast(pl.Int32), pl.col("anime_idx").cast(pl.Int32),
                 pl.col("score").cast(pl.Int8)) \
            .write_parquet(out / f"eval_queries_{split}.parquet")
        counts[split] = q.height
    if not final_exam:
        (out / "eval_queries_test_cold.parquet").unlink(missing_ok=True)
    shutil.copyfile(config.TRAIN_DATA / "eval_seen.parquet", out / "eval_seen.parquet")
    (pl.scan_parquet(config.TRAIN_DATA / "users.parquet")
        .select("user_idx", "gender_id", "joined_bucket",
                "history_ids", "history_scores", "hard_neg_ids")
        .sink_parquet(out / "users_history.parquet"))
    return counts


def export_user_split(out: Path) -> pl.DataFrame:
    split = pl.read_parquet(config.TRAIN_DATA / "_user_split.parquet").select(
        "username", "user_idx", "split")
    split.write_parquet(out / "user_split.parquet")
    return split


def write_contract(out: Path, ckpt: dict, vecs: np.ndarray, user_sd: dict,
                   split: pl.DataFrame, q_counts: dict[str, int]) -> None:
    cfg = ckpt["cfg"]
    metrics = "\n".join(f"  - {k}: {v}" for k, v in ckpt.get("metrics", {}).items())
    user_keys = "\n".join(f"  - `{k}`: {tuple(v.shape)}" for k, v in user_sd.items())
    counts = dict(zip(*[c.to_list() for c in split.group_by("split").len().get_columns()]))
    # The ranker's data_prep reads the first lines containing "Generated" and "Source checkpoint":
    # keep those two lines as they are and both phrases out of every other line.
    text = f"""# export/ CONTRACT

Written by `model/retriever/export.py`: **do not edit by hand**. Re-run the export to update it.

- Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}
- Source checkpoint: epoch={ckpt.get("epoch")}, step={ckpt.get("step")}
- Embedding dim d = {cfg.d} | history_pool = `{cfg.history_pool}` | score_pool = `{cfg.score_pool}` | history_source = `{cfg.history_source}` | use_item_id = {cfg.use_item_id}

Checkpoint metrics (warm slice, val; mask = seen - query, history cap {cfg.eval_history_cap}):
{metrics or "  (n/a)"}

## Files

### `item_vectors.npy`
- shape `{vecs.shape}`, dtype `{vecs.dtype}`, L2-normalized (cosine = dot).
- **row index == anime_idx** (0 = PAD, 1 = OOV, both neutral; real anime 2..N-1).
- **Rows of cold items (`is_cold` in item_index) are encoded with id -> OOV**: the ids of H were
  never trained (H is kept out of every training path), so encoding the real id is noise; OOV
  (real content) matches the serve path measured at cold eval.
- Two uses: (a) brute-force cosine candidates, (b) the history pooling table of UserTower
  (when `history_source = cache`).
- Consumers keep `anime_idx >= 2` when building candidates.

### `item_index.parquet`
- columns `anime_idx: int32` (0..N-1, matches the item_vectors rows), `mal_id: int64` (-1 for
  PAD/OOV), `is_cold: bool` (item in the cold set H; its item_vectors row is content-only).

### `user_tower.pt`
- `state_dict` (user side only, no `item_tower.*`):
{user_keys}
- `d`, `mlp_hidden`, `history_pool`, `score_pool`, `history_source`, `eval_history_cap`,
  `user_features` (vocab/dim), `special_idx`.
- Service: `import UserTower` + the `pool_history` logic. History pools over `item_vectors.npy`
  (`history_source=cache`) or the `hist_emb.*` table in the state_dict (`embed`).
  Input history: the user's MAL list sorted by score desc, cut at `eval_history_cap`.

### `user_split.parquet`
- columns `username`, `user_idx: int32`, `split` in {{train,val,test}}.
- Counts: {counts}

### Eval protocol (the ranker's two-stage eval must score exactly the retriever's protocol)
- Protocol: mask = `seen(user) - the query being scored` (never seen as is: queries are a subset
  of seen); metrics are mean-per-user recall@K / ndcg@K (binary relevance, IDCG normalized by
  `min(R,K)`); the history used to encode U is the `eval_history_cap` prefix of the full list
  (sorted by score desc).
- `eval_queries_val.parquet` / `eval_queries_test.parquet` / `eval_queries_val_cold.parquet`:
  `user_idx: int32, anime_idx: int32, score: int8`, the held-out positives (queries). `score` is
  the query item's original score (0 = completed, unrated); the ranker uses it for the liked
  metric (liked = score >= 1 and score > the user's u_mean). Rows: {q_counts}
- `eval_seen.parquet`: `user_idx, seen_ids: list[int32]`, every interaction of every status
  (plan-to-watch included) of the eval users; the only source for the seen-mask.
- `users_history.parquet`: `user_idx, gender_id, joined_bucket, history_ids, history_scores,
  hard_neg_ids` for every user. Full history sorted (score desc, ties asc); for eval users it is
  the support (queries and cold H removed); hard_neg = dropped + scores 1-4 (at most 64). The
  ranker builds U and its train data from here.
- `eval_queries_test_cold.parquet`: **final exam**, present only after
  `export.py --final-exam` (scored once, when the pipeline is final); a default export deletes it.

## Firewall
The retriever writes here; the ranker and the service only read these files and import model
*definitions*, never training code.
"""
    (out / "CONTRACT.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the Two-Tower bundle -> export/")
    ap.add_argument("--ckpt", type=Path, default=config.RUNS / "best.pt")
    ap.add_argument("--out", type=Path, default=paths.EXPORT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--final-exam", action="store_true",
                    help="also export eval_queries_test_cold.parquet (scored once, when the pipeline is final)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = checkpoint.load(args.ckpt)

    model, spec = build_model(ckpt, args.device)
    num_items = model.item_table.num_items

    vecs = export_item_vectors(model, args.out)
    export_item_index(num_items, args.out)
    user_sd = export_user_tower(ckpt, spec, args.out)
    split = export_user_split(args.out)
    q_counts = export_eval_protocol(args.out, args.final_exam)
    write_contract(args.out, ckpt, vecs, user_sd, split, q_counts)

    print(f"Exported {num_items:,} item vectors {vecs.shape} -> {args.out}")
    files = ["item_vectors.npy", "item_index.parquet", "user_tower.pt", "user_split.parquet",
             "eval_seen.parquet", "users_history.parquet", "CONTRACT.md"]
    files += [f"eval_queries_{s}.parquet" for s in q_counts]
    for f in files:
        print(f"  - {args.out / f}")
    if args.final_exam:
        print("WARNING FINAL EXAM: eval_queries_test_cold.parquet exported; score it once, then "
              "re-run a normal export to delete it.")


if __name__ == "__main__":
    main()

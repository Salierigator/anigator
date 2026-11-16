#!/usr/bin/env python3
"""Build public/anime-index.json: the client-side search index of the guest picker.

Reads data/raw/details.jsonl.gz (titles incl. title_english + synonyms, poster) and
export/item_index.parquet (for the in_corpus flag); read-only on both.

Re-run whenever export/ is re-exported: the index defines what can be picked, so it must follow
the model version, not the crawl version.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DETAILS = ROOT / "data" / "raw" / "details.jsonl.gz"
ITEM_INDEX = ROOT / "export" / "item_index.parquet"
OUT = ROOT / "service" / "frontend" / "public" / "anime-index.json"

POSTER_PREFIX = "https://cdn.myanimelist.net/images/anime/"


def main() -> None:
    corpus = set(pd.read_parquet(ITEM_INDEX)["mal_id"].tolist())

    rows, seen, dropped_nsfw = [], set(), 0
    with gzip.open(DETAILS, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            mal_id = r.get("mal_id")
            if mal_id is None or mal_id in seen:
                continue
            seen.add(mal_id)

            # SFW: matches the backend's nsfw_idx mask (genre Hentai / rating Rx); the service drops
            # these from the results, so they cannot be picked either.
            genres = {g["name"] for g in (r.get("genres") or []) + (r.get("explicit_genres") or [])}
            if "Hentai" in genres or (r.get("rating") or "").startswith("Rx"):
                dropped_nsfw += 1
                continue

            poster = ((r.get("images") or {}).get("jpg") or {}).get("image_url") or ""
            if poster.startswith(POSTER_PREFIX) and poster.endswith(".jpg"):
                poster = poster[len(POSTER_PREFIX):-4]     # "1223/96541", rebuilt by the client
            else:
                poster = ""

            rows.append([
                mal_id,
                r.get("title") or "",
                r.get("title_english") or "",
                [s for s in (r.get("title_synonyms") or []) if s],
                r.get("type") or "",
                r.get("year") or 0,
                round(r["score"], 2) if r.get("score") else 0,
                r.get("members") or 0,
                poster,
                1 if mal_id in corpus else 0,
            ])

    rows.sort(key=lambda r: -r[7])                          # members desc: compresses better
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

    in_corpus = sum(r[9] for r in rows)
    print(f"{OUT.relative_to(ROOT)}: {len(rows)} anime "
          f"({in_corpus} in_corpus, {len(rows) - in_corpus} outside) "
          f"| dropped {dropped_nsfw} nsfw | {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

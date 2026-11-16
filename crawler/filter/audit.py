"""Step 3: measure whether the repair actually worked, instead of assuming it.

Four checks on the output of classify.py:

1. cap        no row above 20 favorites (MAL's supporter cap); rows above 10 are
              supporters and must stay rare: a pile of them means we under-cut
2. residue    no row (outside the re-crawl queue) still ends in something that
              looks like a mined footer: the leftover-dirt check
3. shape      favorites-count distribution of the repaired rows vs the rows that
              were never contaminated: they should look alike, except tier A
              which is pinned at the cap by construction
4. own-list   share of favorites that appear in that user's own animelist.
              Repaired rows must match the never-contaminated baseline: lower =
              footer junk still inside, much higher = we cut real favorites off

Reads classify.py's outputs and mine_footer.py's windows; only prints. Check 4
streams ratings with polars on a user sample and never collects the full table.
It prefers `data/raw/ratings/*.parquet` (column pruning keeps the scan cheap); a
.csv path still works.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import _common as C

ROOT = Path(__file__).resolve().parents[2]


def own_list_rate(ratings_path, groups, sample, seed=0):
    """group -> share of favorites that sit in the user's own animelist."""
    import polars as pl
    rng = random.Random(seed)
    picked = {g: rng.sample(list(rows), min(sample, len(rows))) for g, rows in groups.items()}
    users = {u for rows in picked.values() for u, _ in rows}
    if not users:
        return {}
    scan = (pl.scan_parquet(ratings_path) if str(ratings_path).endswith(".parquet")
            else pl.scan_csv(ratings_path, schema_overrides={"username": pl.String,
                                                             "anime_id": pl.String}))
    lists = (scan
             .select("username", "anime_id")
             .filter(pl.col("username").is_in(list(users)))
             .collect(engine="streaming"))
    own = {}
    for u, a in lists.iter_rows():
        own.setdefault(u, set()).add(a)
    out = {}
    for g, rows in picked.items():
        hit = tot = 0
        for u, favs in rows:
            if u not in own:
                continue
            tot += len(favs)
            hit += sum(1 for x in favs if x in own[u])
        out[g] = (hit / tot if tot else float("nan"), tot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=str(ROOT / "data/raw/profiles.fixed.csv"))
    ap.add_argument("--report", default=str(ROOT / "data/raw/_provenance/classify_report.csv"))
    ap.add_argument("--windows", default=str(ROOT / "data/raw/_provenance/footer_windows.json"))
    ap.add_argument("--ratings", default=str(ROOT / "data/raw/ratings/*.parquet"),
                    help='"" to skip the own-list check')
    ap.add_argument("--sample", type=int, default=3000, help="users per group for check 4")
    args = ap.parse_args()

    _, rows = C.read_profiles(args.profiles)
    tier = {r["username"]: r["tier"] for r in csv.DictReader(open(args.report, encoding="utf-8"))}
    all_F = [w["F"] for w in json.loads(Path(args.windows).read_text()) if w["F"]]

    groups = {}
    for r in rows:
        favs = C.favs_of(r)
        if favs:
            groups.setdefault(tier.get(r["username"], "CLEAN"), []).append((r["username"], favs))

    print("1. cap")
    over20 = [(u, f) for rs in groups.values() for u, f in rs if len(f) > 20]
    over10 = sum(1 for rs in groups.values() for _, f in rs if len(f) > 10)
    n_all = sum(len(v) for v in groups.values())
    print(f"   >20 favorites: {len(over20)} (must be 0)   >10 (supporter): "
          f"{over10} ({over10 / max(n_all, 1):.2%})")
    for u, f in over20[:5]:
        print(f"     [!] {u}: {len(f)} favorites")

    print("2. residue")
    for g, rs in sorted(groups.items()):
        n = sum(1 for _, f in rs if C.suspicious_tail(f, all_F))
        note = "  <- needs a look" if n and g in ("CLEAN", "A") else ""
        print(f"   {g:<6} {n:>7}/{len(rs):<7} rows still ending like a footer{note}")

    print("3. shape")
    for g, rs in sorted(groups.items()):
        lens = [len(f) for _, f in rs]
        at_cap = sum(1 for x in lens if x in C.CAPS) / len(lens)
        print(f"   {g:<6} n={len(rs):>7}  mean={statistics.mean(lens):5.2f}  "
              f"median={statistics.median(lens):4.1f}  at cap={at_cap:6.1%}")

    if args.ratings:
        print("4. own-list (streamed ratings, user sample)")
        for g, (rate, tot) in sorted(own_list_rate(args.ratings, groups, args.sample).items()):
            print(f"   {g:<6} {rate:6.1%} of {tot} favorites are on the user's own list")
        print("   -> A/B/C should be close to CLEAN; "
              "lower = junk left, much higher = real favorites cut")


if __name__ == "__main__":
    main()

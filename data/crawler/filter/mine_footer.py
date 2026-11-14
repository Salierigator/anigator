"""Step 1 — recover the footer ranking sequence F for every slice of the crawl.

F = the ids MAL's site-wide footer (Top Anime 5 + Top Airing 5, deduped) showed
while a given batch of profiles was crawled. It changes over days, so it is
mined per time window using `profile_at` from the crawl state db.

How: a contaminated row ends with F minus whatever the user already favorited.
Most users overlap F in nothing, so the *whole* F shows up verbatim as the tail
of thousands of rows in the same window, while any longer tail (F plus a real
favorite) is essentially unique. Take the longest tail with enough support.

    venv/bin/python data/crawler/filter/mine_footer.py
    venv/bin/python data/crawler/filter/mine_footer.py --window-hours 2 --min-support 30

Read the printed table before trusting the output: windows whose F disagrees
with their neighbours are footer transitions (fine, classify.py handles them by
testing neighbouring windows too), but a window with tiny support or an F that
looks nothing like a top-anime list means the mining failed there.
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import timedelta
from pathlib import Path

import _common as C

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data" / "crawler").is_dir())


def mine_window(rows_favs, min_support, max_len):
    """-> (F, support) for one window, or (None, 0)."""
    best, best_n = None, 0
    counts = collections.Counter()
    for favs in rows_favs:
        for k in range(3, min(max_len, len(favs)) + 1):
            counts[tuple(favs[-k:])] += 1
    for tail, n in counts.items():
        if n >= min_support and (best is None or len(tail) > len(best)
                                 or (len(tail) == len(best) and n > best_n)):
            best, best_n = tail, n
    return (list(best) if best else None), best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=str(ROOT / "data/raw/profiles.csv"))
    ap.add_argument("--db", default=str(ROOT / "data/raw/crawl_state.sqlite"))
    ap.add_argument("--out", default=str(ROOT / "data/raw/_provenance/footer_windows.json"))
    ap.add_argument("--window-hours", type=float, default=2.0)
    ap.add_argument("--min-support", type=int, default=0,
                    help="0 = auto: max(20, 1%% of the rows in the window)")
    ap.add_argument("--max-len", type=int, default=14, help="longest tail considered")
    args = ap.parse_args()

    _, rows = C.read_profiles(args.profiles)
    times = C.row_times(rows, C.read_crawl_times(args.db))
    t0 = C.parse_ts(next(t for t in times if t))
    buckets = collections.defaultdict(list)
    for r, t in zip(rows, times):
        favs = C.favs_of(r)
        if favs:
            buckets[C.window_index(t, t0, args.window_hours)].append(favs)

    windows = []
    for w in sorted(buckets):
        favs = buckets[w]
        thr = args.min_support or max(20, len(favs) // 100)
        F, n = mine_window(favs, thr, args.max_len)
        start = t0 + timedelta(hours=args.window_hours * w)
        windows.append({"window": w, "start": start.isoformat(), "n_rows": len(favs),
                        "min_support": thr, "support": n, "F": F})

    print(f"{'window':>6} {'start':<20} {'rows':>7} {'support':>8}  F")
    prev = None
    for wd in windows:
        F = wd["F"]
        mark = "" if F is None or F == prev else "  <- footer doi"
        print(f'{wd["window"]:>6} {wd["start"][:16]:<20} {wd["n_rows"]:>7} '
              f'{wd["support"]:>8}  {" ".join(F) if F else "(khong mine duoc)"}{mark}')
        prev = F if F else prev

    n_none = sum(1 for w in windows if not w["F"])
    Path(args.out).write_text(json.dumps(windows, indent=1), encoding="utf-8")
    print(f"\n[done] {len(windows)} windows ({n_none} khong mine duoc) -> {args.out}")
    if n_none:
        print("    -> cac window do se duoc classify.py xu ly bang F cua window ke ben; "
              "neu nhieu, ha --window-hours hoac --min-support")


if __name__ == "__main__":
    main()

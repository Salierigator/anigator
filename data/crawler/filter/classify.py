"""Step 2 — cut the footer junk off `favorites_anime`, and say which rows are not safe.

Every row is tested against the F of its own window and of the two neighbouring
windows (footer transitions land mid-window). Outcome per row:

  CLEAN   no F explains the row      -> untouched
  A       cut is certain             -> cut, no re-crawl
          p_min hits the MAL cap (10, or 20 for supporters), so the user cannot
          have had one more favorite: p_min is the real boundary
  B       cut is +-1 uncertain       -> cut to p_min, AND queued for re-crawl
          p_min < cap, so the last kept id might in truth be the user's own
          (a favorite equal to F[0] is deduped away from the junk and becomes
          indistinguishable from it) — only a re-crawl settles it
  C       cannot decide              -> row left as crawled, queued for re-crawl
          neighbouring windows disagree, or no F explains the row but its tail
          sits inside some F in order (= we failed to mine that footer)

Outputs (nothing is overwritten in place):
  profiles.fixed.csv    same 17 columns -> append-compatible with the raw file
  recrawl_users.txt     feed to: crawl_profiles.py --requeue-file <file>
  classify_report.csv   one line per touched row: tier, p_min, before/after

    venv/bin/python data/crawler/filter/classify.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import _common as C

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data" / "crawler").is_dir())


def decide(favs, F_own, F_near, all_F):
    """-> (tier, p_min or None, note)."""
    ps = {C.split_point(favs, F) for F in F_near} if F_near else set()
    ps.discard(None)
    if not ps:
        tail = C.suspicious_tail(favs, all_F)
        if tail:
            return "C", None, f"khong F nao giai thich, duoi nghi ngo: {' '.join(tail)}"
        return "CLEAN", None, ""
    if len(ps) > 1:
        return "C", min(ps), f"cac window lan can cho p khac nhau: {sorted(ps)}"
    p = ps.pop()
    if F_own is None:
        return "C", p, "window nay khong mine duoc F"
    if p in C.CAPS:
        return "A", p, ""
    return "B", p, f"p_min={p} < cap, co the lech +1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=str(ROOT / "data/raw/profiles.csv"))
    ap.add_argument("--db", default=str(ROOT / "data/raw/crawl_state.sqlite"))
    ap.add_argument("--windows", default=str(ROOT / "data/raw/_provenance/footer_windows.json"))
    ap.add_argument("--out", default=str(ROOT / "data/raw/profiles.fixed.csv"))
    ap.add_argument("--recrawl", default=str(ROOT / "data/raw/_provenance/recrawl_users.txt"))
    ap.add_argument("--report", default=str(ROOT / "data/raw/_provenance/classify_report.csv"))
    ap.add_argument("--window-hours", type=float, default=2.0, help="must match mine_footer.py")
    args = ap.parse_args()

    header, rows = C.read_profiles(args.profiles)
    times = C.row_times(rows, C.read_crawl_times(args.db))
    t0 = C.parse_ts(next(t for t in times if t))
    wins = {w["window"]: w["F"] for w in json.loads(Path(args.windows).read_text())}
    all_F = [F for F in wins.values() if F]

    tiers = collections.Counter()
    recrawl, report = [], []
    for r, t in zip(rows, times):
        favs = C.favs_of(r)
        if not favs:
            continue
        w = C.window_index(t, t0, args.window_hours)
        F_own = wins.get(w)
        F_near = [F for F in (wins.get(w - 1), F_own, wins.get(w + 1)) if F]
        F_near = [F for i, F in enumerate(F_near) if F not in F_near[:i]]
        tier, p, note = decide(favs, F_own, F_near, all_F)
        tiers[tier] += 1
        if tier == "CLEAN":
            continue
        if tier in ("A", "B"):
            r["favorites_anime"] = " ".join(favs[:p])
        if tier in ("B", "C"):
            recrawl.append(r["username"])
        report.append({"username": r["username"], "tier": tier, "p_min": p,
                       "n_before": len(favs), "n_after": len(C.favs_of(r)),
                       "before": " ".join(favs), "note": note})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    Path(args.recrawl).write_text("\n".join(dict.fromkeys(recrawl)) + "\n", encoding="utf-8")
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["username", "tier", "p_min", "n_before",
                                          "n_after", "before", "note"])
        w.writeheader()
        w.writerows(report)

    n_favs = sum(1 for r in rows if C.favs_of(r))
    print(f"rows co favorites: {n_favs}")
    for tier, label in (("CLEAN", "sach, khong dung toi"),
                        ("A", "cat chac chan (p_min = cap)"),
                        ("B", "cat nhung +-1 -> re-crawl"),
                        ("C", "khong quyet duoc -> re-crawl (giu nguyen)")):
        n = tiers[tier]
        print(f"  {tier:<6} {n:>8} ({n / max(n_favs, 1):6.1%})  {label}")
    print(f"\n[done] {args.out}")
    print(f"       {len(set(recrawl))} user can crawl lai -> {args.recrawl}")
    print(f"       chi tiet -> {args.report}")


if __name__ == "__main__":
    main()

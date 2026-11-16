"""CLI inference end to end: a MAL username (or a file of mal_ids) -> the two recommendation sections.

A thin entry point: the serving logic lives in the architecture picked with --model (a folder of
model/, reached through model.load), the MAL client in app/clients/mal_api.py. The FastAPI app
uses the same recommender through app/services/real_service.py.

History source, in order:
  1. a username the recommender knows from its own data (offline, no API; for val/test users the
     history is the support, so hits on held-out queries are a good sign).
  2. --live, or a username outside the dataset -> MAL v2 animelist + Jikan profile (needs
     MAL_CLIENT_ID in service/.env), cached as JSON in service/backend/cache/.
  3. --mal-ids <file>: one mal_id per line, treated as completed without a score.

--dump writes the results to service/backend/cache/<name>_recs.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # service/backend/, for `import app`
import app.config  # noqa: E402,F401  (puts the repo root on sys.path)
import model                                              # noqa: E402
from model.base import UserHistory                        # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def fetch_live(username: str, no_cache: bool):
    """MAL v2 animelist + Jikan profile, cached as JSON in service/backend/cache/."""
    from app.clients import mal_api
    CACHE.mkdir(exist_ok=True)
    al_p, pr_p = CACHE / f"{username}_animelist.json", CACHE / f"{username}_profile.json"
    if not no_cache and al_p.exists() and pr_p.exists():
        print(f"[cache] {al_p.name} + {pr_p.name}")
        return json.loads(al_p.read_text()), json.loads(pr_p.read_text())
    print(f"[api] fetch animelist + profile '{username}' ...")
    animelist = mal_api.get_user_anime_list(username)
    profile = mal_api.get_user_profile(username)
    al_p.write_text(json.dumps(animelist, ensure_ascii=False))
    pr_p.write_text(json.dumps(profile or {}, ensure_ascii=False))
    return animelist, profile


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n-- {title} " + "-" * max(0, 76 - len(title)))
    print(f"{'#':>3}  {'title':<46} {'year':>5} {'type':<7} {'MAL':>5}"
          f"{'pred':>8} {'cos':>7}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {(r['title'] or '?')[:46]:<46} {r['year'] or '-':>5} "
              f"{r['type']:<7} {r['mal_score'] if r['mal_score'] is not None else '-':>5}"
              f"{r.get('pred', ''):>8} {r.get('cos', ''):>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend anime for one user (CLI).")
    ap.add_argument("username", nargs="?", help="MAL username (dataset or live)")
    ap.add_argument("--mal-ids", type=Path, help="file of mal_ids (one per line) instead of a username")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cold-k", type=int, default=10)
    ap.add_argument("--anchor", type=int, help="mal_id X: anime like X (still personalized)")
    ap.add_argument("--allow-nsfw", action="store_true", help="do not filter hentai (filtered by default)")
    ap.add_argument("--live", action="store_true", help="fetch from the API even if the username is in the dataset")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dump", action="store_true", help="write the results as JSON to service/backend/cache/")
    ap.add_argument("--model", default="twotower", help="architecture under model/ (default: twotower)")
    args = ap.parse_args()
    if not args.username and not args.mal_ids:
        ap.error("a username or --mal-ids is required")

    t0 = time.time()
    rec = model.load(args.model)
    print(f"[load] {args.model} ready ({time.time() - t0:.1f}s)")

    if args.mal_ids:
        ids = [int(l) for l in args.mal_ids.read_text().split() if l.strip()]
        user, name = UserHistory(mal_ids=ids), args.mal_ids.stem
        print(f"[user] {len(ids)} mal_ids")
    else:
        name = args.username
        user = UserHistory(username=name)
        if args.live or not rec.knows_user(name):
            user.animelist, user.profile = fetch_live(name, args.no_cache)
            if not user.animelist:
                print("[!] empty animelist (private list, unknown user, or no MAL_CLIENT_ID "
                      "in service/.env)")
                user.animelist = []

    t1 = time.time()
    try:
        out = rec.recommend(user, top_k=args.top_k, cold_k=args.cold_k,
                            sfw=not args.allow_nsfw, anchor=args.anchor)
    except KeyError:
        ap.error(f"--anchor {args.anchor} is not in the corpus")
    m = out.meta
    print(f"[user] '{name}' " + " ".join(f"{k}={v}" for k, v in m.items() if k != "history_top")
          + ("   [!] COLD START -> popularity-leaning results" if not m.get("history_count") else ""))
    for h in m.get("history_top", []):
        print(f"        hist: {h['title'][:40]:<40} (score {h['score']})")
    print(f"[infer] {time.time() - t1:.2f}s"
          + (f"  [anchor mal_id={args.anchor}]" if args.anchor else ""))
    for section, rows in out.sections.items():
        title = f"{section} (top {len(rows)})"
        if section == "main" and args.anchor:
            title = f"{section}: anime like mal_id {args.anchor} (top {len(rows)})"
        print_table(title, rows)

    if args.dump:
        CACHE.mkdir(exist_ok=True)
        p = CACHE / f"{name}_recs.json"
        p.write_text(json.dumps({"user": name, **out.sections}, ensure_ascii=False, indent=2,
                                default=str))
        print(f"\n[dump] {p}")


if __name__ == "__main__":
    main()

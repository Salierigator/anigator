"""Crawl user profiles: Jikan primary, full-parity HTML fallback.

FIXED version of data/crawler/crawl_profiles.py — see NOTE.md for what was wrong
and why the fix looks like this. Column layout is byte-for-byte the same as the
old script, so fixed rows can be appended straight onto the existing
data/raw/profiles.csv (cleaning dedups by username keep-last).

Two sources produce IDENTICAL columns (the `source` column records which):
- Jikan /users/{u}/full — primary, because it keeps profile traffic off our IP:
  users.php polling and the ratings crawler already hit MAL directly, and with
  everything running in parallel for days MAL would otherwise see one IP doing
  all of it. Through Jikan, MAL sees Jikan's servers.
- Direct parse of myanimelist.net/profile/{u} — fallback with the same fields
  (sidebar, anime-stats block, favorites are all server-rendered). Jikan proxies
  MAL live and mass-504s during outages (observed for hours), so a circuit
  breaker skips Jikan for 5 min after 10 consecutive failures instead of paying
  its retries on every user.

Only crawls users whose ratings crawl succeeded (`ratings_status='ok'`) — a
private/deleted/empty-list user has no training value, so their profile request
would be pure waste. Run crawl_ratings ahead of (or alongside) this script.

Output: data/raw/profiles.csv. `favorites_anime` = space-separated mal_ids,
`joined`/`birthday` normalized to YYYY-MM-DD. Same resume/dedup semantics as
crawl_ratings: rows append first, state marks second.

    venv/bin/python data/crawler/crawl_profiles.py               # all pending users
    venv/bin/python data/crawler/crawl_profiles.py --limit 5     # smoke test
    venv/bin/python data/crawler/crawl_profiles.py --source html # force one path
    venv/bin/python data/crawler/crawl_profiles.py --requeue-file recrawl.txt
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime

import common

OUT = common.RAW / "profiles.csv"
STAT_KEYS = ["days_watched", "mean_score", "watching", "completed", "on_hold", "dropped",
             "plan_to_watch", "total_entries", "rewatched", "episodes_watched"]
COLS = ["username", "gender", "birthday", "location", "joined", "favorites_anime",
        *STAT_KEYS, "source"]

BREAKER_FAILS = 10      # consecutive Jikan failures before we
BREAKER_PAUSE = 300     # stop trying it for this many seconds (auto mode)

# --- HTML parsing (all server-rendered on the profile page) -----------------
SIDEBAR_RE = (r'<span class="user-status-title[^"]*">{label}</span>'
              r'<span class="user-status-data[^"]*">([^<]+)</span>')
STATS_BLOCK_RE = re.compile(r'class="stats anime"(.*?)class="stats manga"', re.S)
DAYS_RE = re.compile(r'Days:\s*</span>\s*([\d.,]+)')
MEAN_RE = re.compile(r'Mean Score:\s*</span>\s*<span[^>]*>([\d.,]+)</span>')
COUNT_LABELS = ["Watching", "Completed", "On-Hold", "Dropped", "Plan to Watch",
                "Total Entries", "Rewatched", "Episodes"]  # order matches STAT_KEYS[2:]
# favorites = the <ul class="fav-slide"> that lives inside #anime_favorites, so
# the block ends at its OWN </ul>. Never look ahead for the *next* section: the
# old regex did, and for users who set only anime favorites there is no next
# section, so it ran to EOF and swallowed MAL's site-wide footer ranking.
FAV_BLOCK_RE = re.compile(r'id="anime_favorites"(.*?)</ul>', re.S)
# MAL prints the true count in the header right above the block — free checksum
FAV_COUNT_RE = re.compile(r'<h5>\s*Anime\s*\((\d+)\)\s*</h5>\s*<div[^>]*id="anime_favorites"', re.S)
# href anchor keeps CDN poster paths out: cdn.../images/anime/{dir}/{file}.jpg
# also contains "/anime/<digits>/" but never this href prefix
FAV_LINK_RE = re.compile(r'href="https://myanimelist\.net/anime/(\d+)/')


def _date_iso(val):
    """'Nov 28, 2022' -> '2022-11-28'. Partial dates (no year / year only) are
    kept verbatim — MAL lets users hide the year, that is data, not a parse miss."""
    if not val:
        return ""
    try:
        return datetime.strptime(val, "%b %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return val


def _count(block, label):
    # ">Label</a><span>123</span>" (status rows are links) or "Label</span><span>123</span>"
    m = (re.search(r'>' + re.escape(label) + r'</a>\s*<span[^>]*>([\d.,]+)</span>', block)
         or re.search(re.escape(label) + r'</span>\s*<span[^>]*>([\d.,]+)</span>', block))
    return m.group(1).replace(",", "") if m else ""


def favorites(html):
    """-> (ids, ok). ok=False means the page did not parse as expected and the
    row must NOT be trusted; caller marks the user for a retry instead of
    writing a silently wrong favorites column."""
    count = FAV_COUNT_RE.search(html)
    block = FAV_BLOCK_RE.search(html)
    if block is None:
        # no anime favorites set -> MAL omits both the header and the block
        return [], count is None
    favs = list(dict.fromkeys(FAV_LINK_RE.findall(block.group(1))))
    if count is None:
        return favs, False
    return favs, len(favs) == int(count.group(1))


def from_html(html, username):
    """-> row (list) or None when the page did not parse as expected."""
    def side(label):
        m = re.search(SIDEBAR_RE.format(label=label), html)
        return m.group(1).strip() if m else ""

    favs, ok = favorites(html)
    if not ok:
        print(f"    [!] {username}: favorites block did not match its own count — "
              f"MAL layout likely changed, NOT writing this row (check the parser)")
        return None
    m = STATS_BLOCK_RE.search(html)
    block = m.group(1) if m else ""
    days = DAYS_RE.search(block)
    mean = MEAN_RE.search(block)
    return [username, side("Gender"), _date_iso(side("Birthday")), side("Location"),
            _date_iso(side("Joined")), " ".join(favs),
            days.group(1).replace(",", "") if days else "",
            mean.group(1) if mean else "",
            *(_count(block, lbl) for lbl in COUNT_LABELS), "html"]


def from_jikan(u, username):
    st = (u.get("statistics") or {}).get("anime") or {}
    favs = [str(a["mal_id"]) for a in (u.get("favorites") or {}).get("anime", [])]
    return [username, u.get("gender") or "", (u.get("birthday") or "")[:10],
            u.get("location") or "", (u.get("joined") or "")[:10], " ".join(favs),
            *(st.get(k, "") for k in STAT_KEYS), "jikan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="crawl at most N users this run")
    ap.add_argument("--source", choices=["auto", "jikan", "html"], default="auto")
    ap.add_argument("--retry-errors", action="store_true", help="re-queue users marked 'error'")
    ap.add_argument("--requeue-file", help="file with one username per line to crawl again "
                                           "(rows append; cleaning keeps the last one)")
    args = ap.parse_args()

    common.install_signals()
    conn = common.db()
    jikan = common.Client(min_interval=1.1, tries=3)
    html = common.Client(min_interval=1.2, headers={"User-Agent": common.BROWSER_UA})

    if args.retry_errors:
        n = conn.execute("UPDATE users SET profile_status=NULL WHERE profile_status='error'").rowcount
        conn.commit()
        print(f"[i] re-queued {n} error users")

    if args.requeue_file:
        names = [ln.strip() for ln in open(args.requeue_file, encoding="utf-8") if ln.strip()]
        n = conn.executemany("UPDATE users SET profile_status=NULL WHERE username=?",
                             [(u,) for u in names]).rowcount
        conn.commit()
        print(f"[i] re-queued {n}/{len(names)} users from {args.requeue_file}")

    # only users whose ratings crawl succeeded: private (403) / deleted (404) /
    # empty-list users have no training value, so spending profile requests on
    # them is waste — run crawl_ratings ahead of (or alongside) this script
    q = ("SELECT username FROM users WHERE profile_status IS NULL "
         "AND ratings_status='ok' ORDER BY discovered_at")
    if args.limit:
        q += f" LIMIT {args.limit}"
    pending = [r[0] for r in conn.execute(q).fetchall()]
    n_blacklisted = conn.execute(
        "SELECT COUNT(*) FROM users WHERE ratings_status IN ('http_403','http_404','empty')"
    ).fetchone()[0]
    print(f"[i] {len(pending)} users to crawl ({n_blacklisted} blacklisted by ratings) -> {OUT}")

    new_file = not OUT.exists()
    n_done = 0
    jikan_fails = 0
    jikan_down_until = 0.0
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLS)
        for username in pending:
            if common.STOP:
                break
            row, status = None, "error"
            try_jikan = (args.source == "jikan"
                         or (args.source == "auto" and time.time() >= jikan_down_until))
            if try_jikan:
                r = jikan.get(f"{common.JIKAN_BASE}/users/{username}/full")
                if r is not None and r.status_code == 200:
                    row, status = from_jikan(r.json()["data"], username), "ok"
                    jikan_fails = 0
                elif r is not None and r.status_code == 404:
                    # Jikan has its own cache/lookup: its 404 is not MAL's answer,
                    # so fall through to HTML instead of blacklisting the user
                    jikan_fails = 0
                else:
                    jikan_fails += 1
                    if args.source == "auto" and jikan_fails >= BREAKER_FAILS:
                        jikan_down_until = time.time() + BREAKER_PAUSE
                        jikan_fails = 0
                        print(f"[!] jikan down {BREAKER_FAILS}x in a row — "
                              f"html-only for {BREAKER_PAUSE}s")
            if row is None and args.source in ("auto", "html"):
                r = html.get(f"https://myanimelist.net/profile/{username}")
                if r is not None and r.status_code == 200:
                    row = from_html(r.text, username)      # None = parse mismatch
                    status = "ok" if row else "error"
                elif r is not None and r.status_code == 404:
                    status = "http_404"
            if row:
                w.writerow(row)
                f.flush()
            conn.execute("UPDATE users SET profile_status=?, profile_at=? WHERE username=?",
                         (status, common.now_iso(), username))
            conn.commit()
            n_done += 1
            tag = status if not row else f"{status} ({row[-1]})"
            print(f"[+] {n_done}/{len(pending)} {username}: {tag}")

    print(f"[done] {n_done} users this run")
    for st, n in conn.execute(
            "SELECT profile_status, COUNT(*) FROM users GROUP BY profile_status"):
        print(f"    {st or 'pending'}: {n}")
    conn.close()


if __name__ == "__main__":
    main()

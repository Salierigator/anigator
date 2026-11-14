"""Shared pieces for the favorites-repair pipeline (see NOTE.md).

The whole pipeline rests on one property of the bug: the junk appended to
`favorites_anime` is not "some footer ids" but exactly

    F minus the ids the user really favorited, in F's own order

because the crawler deduped with dict.fromkeys (first occurrence wins). So with
F known, a split point p is *valid* iff

    row[p:] == [x for x in F if x not in row[:p]]

which is an equality test, not a heuristic.
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timedelta

CAPS = (10, 20)   # MAL favorites cap: 10 normal, 20 for supporters


def read_profiles(path):
    """-> (header, rows) with rows as dicts, in file order (= crawl order)."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def read_crawl_times(db_path):
    """username -> profile_at (ISO). Only the LAST crawl of a user is recorded,
    which is fine: duplicates are rare and windows are hours wide."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {u: t for u, t in conn.execute(
        "SELECT username, profile_at FROM users WHERE profile_at IS NOT NULL")}
    conn.close()
    return out


def row_times(rows, times):
    """Timestamp per row. Rows whose user is missing from the db carry the last
    known timestamp forward — the file is append-only in crawl order, so that is
    accurate to within one row."""
    out, last = [], None
    for r in rows:
        last = times.get(r["username"], last)
        out.append(last)
    if out and out[0] is None:              # nothing known before the first row
        first = next((t for t in out if t), None)
        out = [t or first for t in out]
    return out


def window_index(ts, t0, hours):
    if ts is None:
        return -1
    d = datetime.fromisoformat(ts.replace("Z", "+00:00")) - t0
    return int(d / timedelta(hours=hours))


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def favs_of(row):
    return row["favorites_anime"].split()


def split_point(favs, F):
    """Smallest valid p, or None when the row cannot be explained by F at all
    (= the row is clean, provided F is the right one for its crawl time)."""
    for p in range(len(favs) + 1):
        head = set(favs[:p])
        if favs[p:] == [x for x in F if x not in head]:
            return p
    return None


def is_subsequence(tail, F):
    """tail appears inside F in order (not necessarily contiguous)."""
    it = iter(F)
    return all(x in it for x in tail)


def suspicious_tail(favs, all_F, min_len=3):
    """Safety net for rows no F explains: junk whose exact F we failed to mine.

    Ids absent from every mined F are ignored (that footer had a title we never
    saw); what must hold is that >=min_len of the tail's ids DO appear in one F
    and in that F's order. Deliberately trigger-happy — a false positive only
    costs one re-crawl, a false negative leaves dirt in the data.
    """
    if not all_F:
        return None
    union = set().union(*all_F)
    for k in range(len(favs), min_len - 1, -1):
        tail = favs[-k:]
        known = [x for x in tail if x in union]
        if len(known) >= min_len and any(is_subsequence(known, F) for F in all_F):
            return tail
    return None

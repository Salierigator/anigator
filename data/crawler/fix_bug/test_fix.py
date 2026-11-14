"""Acceptance test for the fixed profile parser — runs offline on saved pages.

    venv/bin/python data/crawler/fix_bug/test_fix.py     # (also works under pytest)

Fixtures are the real MAL pages of 27/07/2026, one per branch of the parser:
journeyGavin (contaminated by the old regex), __Prexos__ (1 favorite, also
contaminated), dom53 (clean: has manga/character favorites after the anime
block), Mircmon (no favorites at all). Expected values were verified by hand
against the live profile pages.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crawl_profiles as cp  # noqa: E402

# username -> (favorites, a few fields that must survive the fix untouched)
EXPECTED = {
    "journeyGavin": (
        "58567 59514 58259 59361 59135 57616 57891 57181 57334 56843",
        {"joined": "2024-10-05", "watching": "23", "completed": "107",
         "total_entries": "297", "mean_score": "8.00"},
    ),
    "__Prexos__": (
        "1735",
        {"gender": "Male", "joined": "2025-04-11", "total_entries": "55"},
    ),
    "dom53": (
        "33674 19815 20785 28171 21 31764 28907 30276 918 4181",
        {"gender": "Male", "birthday": "1998-02-27", "location": "Czech republic",
         "joined": "2015-11-16", "total_entries": "90", "mean_score": "10.00"},
    ),
    "Mircmon": (
        "",
        {"gender": "Male", "birthday": "1994-10-24", "joined": "2014-10-23",
         "total_entries": "1058"},
    ),
}
# what the old regex leaked in: MAL's site-wide footer ranking on 27/07/2026
FOOTER_IDS = {"52991", "61316", "5114", "61469", "9253", "60636", "59193", "21"}


def _row(username):
    html = gzip.open(HERE / "fixtures" / f"{username}.html.gz", "rt", encoding="utf-8").read()
    row = cp.from_html(html, username)
    assert row is not None, f"{username}: parser rejected the page"
    return dict(zip(cp.COLS, row))


def test_favorites_exact():
    for username, (favs, _) in EXPECTED.items():
        got = _row(username)["favorites_anime"]
        assert got == favs, f"{username}: favorites {got!r} != {favs!r}"


def test_no_footer_leak():
    """dom53 genuinely favorites One Piece (21) — only ids beyond the user's own
    favorites count as a leak, which the exact match above already pins down."""
    for username in ("journeyGavin", "__Prexos__"):
        got = set(_row(username)["favorites_anime"].split())
        assert not (got & FOOTER_IDS), f"{username}: footer ids leaked into favorites"


def test_other_fields_unchanged():
    for username, (_, fields) in EXPECTED.items():
        row = _row(username)
        for k, v in fields.items():
            assert row[k] == v, f"{username}.{k}: {row[k]!r} != {v!r}"


def test_count_mismatch_is_rejected():
    """Integrity check must fail loud: drop the last <li> and the row is refused."""
    html = gzip.open(HERE / "fixtures" / "journeyGavin.html.gz", "rt", encoding="utf-8").read()
    i = html.rindex('<li class="btn-fav"', 0, html.index("</ul>", html.index('id="anime_favorites"')))
    broken = html[:i] + html[html.index("</ul>", i):]
    assert cp.from_html(broken, "journeyGavin") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[done] {len(fns)} checks passed")

"""Regression test for the split logic, on rows whose truth we checked by hand.

Ground truth: the four contaminated rows in the 27/07/2026 smoke crawl, verified
against the live MAL profiles. NightLocked in particular really does have 5114
as their 10th favorite, which is what makes "cut everything that looks like a
footer id" wrong and the exact split test right.
"""
from __future__ import annotations

import _common as C
from classify import decide

F = ["52991", "61316", "5114", "61469", "9253", "60636", "59193", "21"]
# same crawl, different day: Top Airing had 50250 instead of 60636
F_OLD = ["52991", "61316", "5114", "61469", "9253", "59193", "21", "50250"]

ROWS = {
    "journeyGavin": "58567 59514 58259 59361 59135 57616 57891 57181 57334 56843 "
                    "52991 61316 5114 61469 9253 60636 59193 21",
    "Warny": "52635 2605 48849 44074 40028 43299 36649 34599 9253 46102 "
             "52991 61316 5114 61469 60636 59193 21",
    "NightLocked": "19815 33352 30484 9253 19221 23847 14813 37450 11061 5114 "
                   "52991 61316 61469 60636 59193 21",
    "Hectabyte": "39535 45576 48316 51179 52991 61316 5114 61469 9253 60636 59193 21",
    "dom53": "33674 19815 20785 28171 21 31764 28907 30276 918 4181",          # clean
    "__Prexos__": "1735 52991 61316 5114 61469 9253 59193 21 50250",           # older footer
}
EXPECT_P = {"journeyGavin": 10, "Warny": 10, "NightLocked": 10, "Hectabyte": 4, "dom53": None}


def test_split_point():
    for user, p in EXPECT_P.items():
        got = C.split_point(ROWS[user].split(), F)
        assert got == p, f"{user}: p_min={got}, expected {p}"


def test_nightlocked_keeps_its_tenth_favorite():
    favs = ROWS["NightLocked"].split()
    kept = favs[:C.split_point(favs, F)]
    assert kept[-1] == "5114" and len(kept) == 10


def test_old_footer_needs_its_own_F():
    favs = ROWS["__Prexos__"].split()
    assert C.split_point(favs, F_OLD) == 1          # right F -> exact
    assert C.split_point(favs, F) is None           # wrong F -> looks clean...
    assert C.suspicious_tail(favs, [F]) is not None  # ...but the safety net fires


def test_clean_row_survives_the_safety_net():
    """dom53 genuinely favorites One Piece; one shared id must not trip anything."""
    assert C.suspicious_tail(ROWS["dom53"].split(), [F]) is None


def test_tiers():
    def tier(user, F_own=F, near=(F,), all_F=(F,)):
        return decide(ROWS[user].split(), F_own, list(near), list(all_F))[0]

    assert tier("journeyGavin") == "A"      # p_min = 10 = cap -> certain
    assert tier("Hectabyte") == "B"         # p_min = 4 < cap  -> +-1, re-crawl
    assert tier("dom53") == "CLEAN"
    assert tier("__Prexos__") == "C"        # wrong F -> undecided, re-crawl
    # no F mined for the row's own window -> not trusted, even if a neighbour's F fits
    assert decide(ROWS["journeyGavin"].split(), None, [F], [F])[0] == "C"
    # neighbouring windows give different cut points -> C
    F1, F2 = ["a", "b", "c"], ["b", "c"]
    assert decide(["x", "a", "b", "c"], F1, [F1, F2], [F1, F2])[0] == "C"


def test_ambiguous_case_is_flagged_not_silently_wrong():
    """User whose real last favorite IS F[0]: p_min under-cuts by exactly 1, and
    the row must land in B (queued for re-crawl), never in A."""
    real = ["1", "2", "3", F[0]]
    row = real + [x for x in F if x not in real]
    p = C.split_point(row, F)
    assert p == len(real) - 1                       # the known +-1 blind spot
    assert decide(row, F, [F], [F])[0] == "B"

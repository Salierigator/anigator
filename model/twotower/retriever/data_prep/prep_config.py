"""Shared config of data_prep: the only source for seed, caps, labels and the cold-item set.

02/03/05/06 import it so the definitions cannot drift between scripts; 06 echoes it into
feature_spec.json so train and serve read back the definitions the data was built with.

To ablate the labeling, edit POS_STATUSES / is_hardneg_expr here and re-run 02 -> 06 (02 changes
n_pos, so split membership changes too).
"""
import polars as pl

SEED = 42
EVAL_MIN_POS = 11        # 03: only users with n_pos >= 11 can go to val/test
EVAL_QUERY_FRAC = 0.2    # share of an eval user's warm positives held out as queries
HARD_NEG_CAP = 64        # hard_neg_ids kept per user
COLD_FRAC = 0.05         # share of the newest anime (by start_date) in the cold set H (01)
EVAL_HISTORY_CAP = 1024  # default history prefix cap at eval (echoed into the spec; the train config can override it)

# --- labels (positive and hard-neg are disjoint by construction: a positive needs a score outside [1,4]) ---
POS_STATUSES = ["completed", "watching"]


def _score():
    return pl.col("score").cast(pl.Int64, strict=False)


def is_pos_expr():
    """positive = status in POS_STATUSES and score not in [1,4] (keeps score 0 and 5..10)."""
    return pl.col("status").is_in(POS_STATUSES) & ~_score().is_between(1, 4)


def is_hardneg_expr():
    """hard-neg = dropped, or score in [1,4] with any status: 'gave up' + 'watched and disliked'."""
    return (pl.col("status") == "dropped") | _score().is_between(1, 4)


# echoed into feature_spec.json (06) as a readable description: change it together with the exprs
LABELS_SPEC = {
    "positive": "status in {completed, watching} & score not in [1,4]",
    "pos_statuses": POS_STATUSES,
    "hard_negative": "status == dropped | score in [1,4] (any status)",
}

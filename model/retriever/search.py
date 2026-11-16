"""Hyperparameter search driver for the retriever (random | grid); importable without torch.

Random search beats grid beyond 3-4 dimensions (Bergstra & Bengio). Two stages: COARSE on
a user subset (train_user_frac) ranks configs cheaply, then the top-K are confirmed on the
full data. Each config gets a deterministic run_name (its non-default knobs, sorted), so
duplicates collapse and a dropped Colab session resumes: every run_name is one run
directory of runs.py.

`runs.py` defines the sweeps and calls `run_search(run_experiment, iter_configs(space, ...))`;
run_experiment writes best.pt/history/row into the run directory, and resume skips runs that
already have row.json.
"""
from __future__ import annotations

import itertools
import random
from typing import Callable, Dict, Iterable, List, Tuple

# Knob abbreviations for short run names (unknown knobs keep their own name).
_ABBR = {
    "d": "d", "use_item_id": "id", "id_dim": "iddim", "id_dropout": "iddrop",
    "history_source": "hist", "train_hist_len": "hl", "history_pool": "hpool",
    "score_pool": "spool", "tau": "tau", "logq_alpha": "alpha", "m_hardneg": "mhn",
    "lr": "lr", "weight_decay": "wd", "optimizer": "opt", "cosine_lr": "cos",
    "batch_size": "bs", "epochs": "ep", "hist_dropout": "hdrop",
    "max_examples_per_user": "cap", "train_user_frac": "uf",
    "use_synopsis": "syn", "synopsis_dim": "sd", "synopsis_proj_hidden": "sph",
    "synopsis_normalize": "snorm",
}

# Dependent knobs: with the parent off the children are dropped, so the same model never
# runs twice under two names.
_DEPENDENT = {
    "use_synopsis": ["synopsis_dim", "synopsis_proj_hidden", "synopsis_normalize"],
    "use_item_id": ["id_dim", "id_dropout"],
}


def canonicalize(overrides: Dict) -> Dict:
    """Drop child knobs whose parent is off, so configs that differ only in a dead knob share a
    run_name and dedupe."""
    ov = dict(overrides)
    for parent, children in _DEPENDENT.items():
        if not ov.get(parent, False):
            for c in children:
                ov.pop(c, None)
    if ov.get("history_pool", "mean") != "mean":   # attn ignores score_pool: one model, one name
        ov.pop("score_pool", None)
    return ov


def _vstr(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}".replace(".", "").replace("-", "m")   # 0.5 -> 05, 1e-5 -> 1em05
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v) if v else "0"
    return str(v).replace(".", "")


def deterministic_run_name(overrides: Dict, prefix: str = "v5") -> str:
    """Stable name from the non-default knobs (sorted by key, so dict order does not matter)."""
    ov = canonicalize(overrides)
    parts = [f"{_ABBR.get(k, k)}{_vstr(v)}" for k, v in sorted(ov.items())]
    return f"{prefix}_" + "_".join(parts) if parts else f"{prefix}_base"


def iter_configs(space: Dict[str, List], method: str = "random", n: int = 20,
                 seed: int = 0, fixed: Dict = None, prefix: str = "v5"
                 ) -> Iterable[Tuple[str, Dict]]:
    """Yield (run_name, overrides) from `space` {knob: [values]}.
    `fixed` applies to every config (e.g. {'train_user_frac': 0.15, 'epochs': 2} for the coarse
    stage). grid = itertools.product (exhaustive); random = n seeded samples. Configs are
    deduped by their canonical run_name, so no identical model runs twice."""
    fixed = fixed or {}
    seen = set()
    if method == "grid":
        keys = list(space)
        for combo in itertools.product(*(space[k] for k in keys)):
            ov = canonicalize({**fixed, **dict(zip(keys, combo))})
            name = deterministic_run_name(ov, prefix)
            if name in seen:
                continue
            seen.add(name)
            yield name, ov
    elif method == "random":
        rng = random.Random(seed)
        tries, cap = 0, max(n * 50, 100)
        while len(seen) < n and tries < cap:
            tries += 1
            ov = canonicalize({**fixed, **{k: rng.choice(v) for k, v in space.items()}})
            name = deterministic_run_name(ov, prefix)
            if name in seen:
                continue
            seen.add(name)
            yield name, ov
    else:
        raise ValueError(f"unsupported method '{method}' (grid|random)")


def run_search(run_experiment_cb: Callable, configs: Iterable[Tuple[str, Dict]],
               exists_fn: Callable[[str], bool] = None, dry_run: bool = False) -> List[Tuple]:
    """Run each config through run_experiment_cb(run_name, **overrides), skipping runs that
    exists_fn reports as done (resume). dry_run only prints the names (to check a space
    locally). Returns [(run_name, overrides, out)]."""
    results = []
    for run_name, overrides in configs:
        if exists_fn is not None and exists_fn(run_name):
            print(f"  skip {run_name} (already done)")
            continue
        print(f"-> {run_name}  {overrides}")
        out = None if dry_run else run_experiment_cb(run_name, **overrides)
        results.append((run_name, overrides, out))
    return results

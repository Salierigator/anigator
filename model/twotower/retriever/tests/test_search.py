"""Search driver invariants: deterministic, order-independent run_name with dependent knobs
canonicalized; random search reproducible and capped; grid exhaustive and deduped; run_search
resumes (skips existing runs). Needs neither train data nor torch."""
from model.twotower.retriever.training import search


def test_run_name_order_independent():
    a = search.deterministic_run_name({"epochs": 2, "train_hist_len": 64})
    b = search.deterministic_run_name({"train_hist_len": 64, "epochs": 2})
    assert a == b, "run_name must not depend on dict order (sorted by key)"


def test_canonicalize_drops_disabled_children():
    c = search.canonicalize({"use_synopsis": False, "synopsis_dim": 64, "epochs": 2})
    assert c == {"use_synopsis": False, "epochs": 2}, "synopsis off -> synopsis_dim dropped"
    # configs that differ only in a dead child knob share a run_name (deduped, never retrained)
    n48 = search.deterministic_run_name({"use_synopsis": False, "synopsis_dim": 48})
    n64 = search.deterministic_run_name({"use_synopsis": False, "synopsis_dim": 64})
    assert n48 == n64
    # synopsis on -> synopsis_dim matters -> different names
    assert (search.deterministic_run_name({"use_synopsis": True, "synopsis_dim": 48})
            != search.deterministic_run_name({"use_synopsis": True, "synopsis_dim": 64}))


def test_random_reproducible_and_capped():
    space = {"train_hist_len": [32, 64, 96], "logq_alpha": [1.0, 0.75], "optimizer": ["adam", "adamw"]}
    n1 = [name for name, _ in search.iter_configs(space, "random", n=5, seed=0)]
    n2 = [name for name, _ in search.iter_configs(space, "random", n=5, seed=0)]
    assert n1 == n2, "same seed -> same config sequence"
    assert len(n1) == 5 and len(set(n1)) == 5, "exactly n configs, no duplicates"


def test_grid_exhaustive_with_fixed_and_dedupe():
    space = {"use_synopsis": [False, True], "synopsis_dim": [48, 64]}
    cfgs = list(search.iter_configs(space, "grid", fixed={"epochs": 2}))
    names = [name for name, _ in cfgs]
    # canonical combos: syn=False (sd dropped -> 1) + syn=True x {48,64} (2) = 3
    assert len(names) == len(set(names)) == 3
    assert all(ov.get("epochs") == 2 for _, ov in cfgs), "fixed applies to every config"


def test_run_search_skips_existing():
    calls = []
    search.run_search(lambda name, **ov: calls.append(name),
                      [("a", {}), ("b", {}), ("c", {})],
                      exists_fn=lambda nm: nm == "b")
    assert calls == ["a", "c"], "must skip the finished run (b) and run a + c (resume)"


def test_run_search_dry_run_no_call():
    calls = []
    res = search.run_search(lambda name, **ov: calls.append(name),
                            [("a", {"epochs": 2})], dry_run=True)
    assert calls == [] and len(res) == 1, "dry_run only prints names, never calls the callback"

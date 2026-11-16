# Cleaning

Turns the crawler's snapshot in `data/raw/` into the training tables in `data/clean/`. Both folders are local and not committed.

**The pipeline is being rewritten** for the 2026 crawl, so there is no code here yet. It will read the raw schemas listed in the [crawler README](../crawler/README.md#output), write `data/clean/`, and refresh `data/samples/`.

## Rules behind the current model's data

The current model was trained on a mid-2025 snapshot, cleaned from 124.3M ratings / 29.0k anime / 337k profiles down to **120.0M ratings / 22.8k anime / 292.6k users** (98.20% sparsity). Every rule is quantitative rather than a hand-picked threshold: drop unknown-status and orphan rows, dedup, four bot rules (implausible rate counts, near-constant scores, impossible watch volume, mass-add non-raters), then iterative k-core (user ≥ 10 ratings, anime ≥ 20) until it converges.

# model/

Recommender architectures, one folder each. `twotower/` (two-tower retriever + LightGBM ranker) is the only one so far.

```
model/
├── base.py       # the contract: UserHistory, Recommendations, Recommender
├── __init__.py   # load(name) -> the serving class of model/<name>/
├── paths.py      # train_dir(name), export_dir(name)
└── twotower/     # one architecture: data prep, training, export, serve.py
```

## What an architecture owns

Each architecture decides for itself how it prepares training data, trains, and what its exported files look like. Three things are fixed:

| | |
|---|---|
| **Folders** | code in `model/<name>/`, training data and checkpoints in `train/<name>/`, the exported bundle in `export/<name>/` |
| **Input** | the shared cleaned data in `data/clean/` |
| **Serving** | `model/<name>/serve.py` has `load()`, returning a [`Recommender`](base.py) that reads only `export/<name>/` |

## The serving contract

`base.py` speaks MAL, not model internals: a `UserHistory` is a MAL animelist (or a bare list of `mal_id`s, or a username the model already knows) and the answer is a `Recommendations` object — named sections of rows (`main` always, others such as `cold` if the architecture has them) plus a few facts about the call.

```python
import model
from model.base import UserHistory

rec = model.load("twotower")
out = rec.recommend(UserHistory(mal_ids=[1, 1535, 9253]), top_k=20)
out.sections["main"]      # rows: mal_id, title, ..., and the model's own scores
```

The backend and the CLI use nothing else, so serving a different architecture is a setting (`RECOMMENDER=<name>` for the API, `--model <name>` for the CLI), not a code change.

## Adding an architecture

1. Create `model/<name>/` with its own prep, training and export code, writing to `train/<name>/` and `export/<name>/`.
2. Add `serve.py` with a `Recommender` subclass and `load()`.
3. Run the service with `RECOMMENDER=<name>`.

The service needs changes only if the new model wants an input `UserHistory` cannot carry or returns something the API has no field for.

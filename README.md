<p align="center">
  <a href="https://salierigator.github.io/anime-recommender/"><img src="images/title_lockup_icon115.png" alt="Anigator" width="440"></a>
</p>

A web app that turns your watch history (or a list of your favorite anime) into a personalized anime recommendation list.

**[▶ Try the live demo ◀](https://salierigator.github.io/anime-recommender/)** — enter your MyAnimeList username for a ranked list.

<sub><i>The backend is on a free tier, so the first request after it's been idle may take ~60s to wake up. Sorry about that.</i></sub> <sub>(,,¬﹏¬,,)</sub>

<p align="center">
  <img src="images/demo.png" alt="Anigator — personalized anime recommendations for a MyAnimeList user" width="820">
</p>

---

## Architecture

Two-stage recommender architecture (retrieval + rerank) — the standard shape for recommenders that must score a large catalog under a latency budget:

1. **Retrieval** — a two-tower model embeds users and anime into one 128-d cosine space: cheap, recall-oriented.
2. **Ranking** — LightGBM (lambdarank) reranks the top 200 candidates on 29 features (user–item cosine, history affinity, genre/theme overlap, MAL metadata, …). Expensive per item, so it only ever sees the shortlist.
3. **Serving** — FastAPI backend + React frontend. Newly released titles the model never trained on (also called cold items) are served as a separate section ranked by cosine, never through the ranker — the ranker is trained on warm items and buries cold ones.

For a detailed description of each stage, see the component README: [retriever](model/twotower/retriever/README.md), [ranker](model/twotower/ranker/README.md), and [service](service/README.md).

```
data/  ──>  model/twotower/  ──>  export/twotower/  ──>  service/
raw         retriever             item_vectors           FastAPI + React
clean       + ranker              user_tower             (read-only consumer)
                                  ranker.txt
```

`export/twotower/` is the contract between training and serving: the retriever and ranker save their trained models here, and the service only reads them. Neither side imports the other's training code, so the serving path cannot silently drift from what was trained.

`model/` holds one folder per recommender architecture; `twotower/` is the one described above and the only one so far. The service reaches a model through a small interface (`model/base.py`) and picks the architecture by name, so another architecture is a new folder rather than a change to the service — see [`model/README.md`](model/README.md).

## Repository layout

| Path | What's in it |
|---|---|
| `crawler/` | MyAnimeList crawler, writes `data/raw/` |
| `cleaning/` | `data/raw/` → `data/clean/` (pipeline being rewritten) |
| `data/` | Schema samples; raw/clean data are not committed |
| `model/base.py` | The interface every architecture's serving class implements (`Recommender`) |
| `model/twotower/retriever/` | Two-tower model: data prep, training, export to `export/twotower/` |
| `model/twotower/ranker/` | LightGBM reranker: pool building, training, eval gate, export to `export/twotower/` |
| `model/twotower/serve.py` | Serving core: retrieve, rerank, cold section |
| `train/<architecture>/` | Training data and checkpoints (not committed) |
| `export/<architecture>/` | Model exports (not committed) |
| `service/backend/` | FastAPI app + CLI (`recommend.py`), read-only consumer of `export/` |
| `service/frontend/` | React + Vite + TypeScript UI |

## Quickstart (Local run)

**Prerequisites:** Python 3.9, Node.js 18+, and (optional, for looking up live MAL usernames) a [MyAnimeList API Client ID](https://myanimelist.net/apiconfig).

```bash
# 1. Clone the repo
git clone https://github.com/Salierigator/anime-recommender.git
cd anime-recommender

# 2. Create a virtual environment and install the Python dependencies
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r service/backend/requirements.txt   # FastAPI + uvicorn for the backend

# 3. (Optional — real mode only) add your MAL Client ID
echo "MAL_CLIENT_ID=your_id_here" > service/.env
```

> **Note:** The run commands below need the trained model files in `export/twotower/` — a download link will be added later. Mock mode runs without them.

```bash
# Backend, mock mode — returns fixtures, loads no model. Best for frontend work.
cd service/backend && MOCK_MODE=1 uvicorn app.main:app --reload --port 8000

# Backend, real mode — needs export/twotower/; MAL_CLIENT_ID in service/.env for live usernames
cd service/backend && MOCK_MODE=0 uvicorn app.main:app --port 8000

# Frontend, run it in another terminal
cd service/frontend && npm install && npm run dev
# Open this link in your browser:  http://localhost:5173/

# CLI — no server; --mal-ids needs no MAL key
venv/bin/python service/backend/recommend.py --mal-ids service/backend/fixtures/dummy_mal_ids.txt
venv/bin/python service/backend/recommend.py <mal-username> --top-k 20
```

## Data & Training

The datasets are MAL-derived and gigabytes in size, so they are not part of this repo; `data/samples/` holds the first rows of each file as a schema reference. The model currently in `export/twotower/` was trained on a mid-2025 snapshot. A fresh 2026 crawl of user ratings and profiles is done (see [`crawler/README.md`](crawler/README.md)); its cleaning pipeline ([`cleaning/`](cleaning/README.md)) is the next step.

Retriever training runs on Colab GPU; the ranker trains locally on CPU. Both write into `export/twotower/`, and the service picks up a new model with no code change — see [`model/twotower/README.md`](model/twotower/README.md) for the full retrain loop.

## History

This project grew out of my graduation thesis. The full thesis version (all experiments, baselines, ablations, docs) is frozen at tag [`thesis-final`](https://github.com/Salierigator/anime-recommender/tree/thesis-final).

I'd be glad if you find this project interesting, open for discussion anytime. Feel free to reach out:
- Email: hieulhp6@gmail.com
- Discord: salierigator

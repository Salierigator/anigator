# data/

Local data, written and read by the code in this repo: **[`crawler/`](../crawler/) → `raw/` → [`cleaning/`](../cleaning/) → `clean/`**.

| Path | Committed | What |
|---|---|---|
| `raw/` | no | Crawler output, the 2026 MAL snapshot: `anime_*.jsonl`, `users.csv`, `ratings/`. |
| `clean/` | no | Training input. Not produced yet: the cleaning pipeline is being rewritten. |
| `samples/` | yes | First rows of each `raw/` table — the schema reference: `anime_jikan`, `anime_mal` (5 anime each), `users` (5 users), `ratings` (2,591 rows). |

The tables are MAL-derived and gigabytes in size, so nothing under `raw/` or `clean/` is committed. Read schemas from `samples/` instead of opening the real files.

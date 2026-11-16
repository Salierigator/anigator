# data/

Local data, written and read by the code in this repo: **[`crawler/`](../crawler/) → `raw/` → [`cleaning/`](../cleaning/) → `clean/`**.

| Path | Committed | What |
|---|---|---|
| `raw/` | no | Crawler output: the 2026 MAL snapshot. |
| `clean/` | no | Training input. Not produced yet: the cleaning pipeline is being rewritten. |
| `samples/` | yes | First rows of each table — the schema reference (`old_*`: the current model's data, `new_*`: the 2026 crawl). |

The tables are MAL-derived and gigabytes in size, so nothing under `raw/` or `clean/` is committed. Read schemas from `samples/` instead of opening the real files.

"""Crawl the full anime catalog from the official MAL API v2 ranking endpoint.

/anime/ranking?ranking_type=all pages through every anime, 500 per request, and returns
the catalog fields cleaning needs, including num_favorites. Genres, themes and
demographics arrive merged in `genres`.

Output: data/raw/anime_mal.jsonl, one MAL anime object per line. A sweep takes about
a minute, so there is no resume: rows go to a .tmp file that replaces the output only
once the last page has been read.
"""
from __future__ import annotations

import argparse
import json

import common

OUT = common.RAW / "anime_mal.jsonl"
FIELDS = ("id,title,alternative_titles,media_type,status,num_episodes,start_date,end_date,"
          "start_season,broadcast,source,rating,nsfw,average_episode_duration,synopsis,"
          "genres,studios,mean,rank,popularity,num_list_users,num_scoring_users,num_favorites,"
          "created_at,updated_at")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int,
                    help="read at most N pages (smoke test: the output file is left untouched)")
    args = ap.parse_args()

    common.install_signals()
    common.RAW.mkdir(parents=True, exist_ok=True)
    client = common.Client(min_interval=0.5,
                           headers={"X-MAL-CLIENT-ID": common.load_mal_client_id()})
    tmp = OUT.with_name(OUT.name + ".tmp")
    url = f"{common.MAL_BASE}/anime/ranking"
    params = {"ranking_type": "all", "limit": 500, "fields": FIELDS, "nsfw": "true"}
    seen = set()
    pages = 0

    with open(tmp, "w", encoding="utf-8") as f:
        while url and not common.STOP and (args.pages is None or pages < args.pages):
            r = client.get(url, params=params)
            if r is None or r.status_code != 200:
                print(f"[-] page {pages + 1}: HTTP {r.status_code if r else 'fail'}, stopping")
                break
            d = r.json()
            for entry in d["data"]:
                node = entry["node"]
                if node["id"] not in seen:  # ranks can shift between two pages
                    seen.add(node["id"])
                    f.write(json.dumps(node, ensure_ascii=False) + "\n")
            pages += 1
            url = d.get("paging", {}).get("next")
            params = None  # paging.next already carries the query string
            if pages % 10 == 0 or args.pages:
                print(f"[+] page {pages} done ({len(seen)} anime)")

    if url is None:
        tmp.replace(OUT)
        print(f"[done] {len(seen)} anime in {pages} pages -> {OUT}")
    else:
        print(f"[partial] {len(seen)} anime in {pages} pages -> {tmp.name}; {OUT.name} left untouched")


if __name__ == "__main__":
    main()

"""Shared plumbing for the MAL crawlers: paths, rate-limited HTTP with backoff, SQLite state.

State lives in data/raw/crawl_state.sqlite (WAL). Rule shared by every crawler: data is
appended first and state is marked second, so a crash between the two re-crawls that
unit on resume and cleaning must dedup.
"""
from __future__ import annotations

import signal
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB_PATH = RAW / "crawl_state.sqlite"

MAL_BASE = "https://api.myanimelist.net/v2"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

STOP = False


def _handler(signum, frame):
    global STOP
    if STOP:  # second Ctrl-C = hard exit
        raise KeyboardInterrupt
    STOP = True
    print("\n[!] stop requested: finishing the current unit, then saving state "
          "(Ctrl-C again to force quit)")


def install_signals():
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_mal_client_id() -> str:
    """MAL_CLIENT_ID from the environment, else from service/.env (shared with the backend)."""
    import os
    cid = os.environ.get("MAL_CLIENT_ID")
    env_path = ROOT / "service" / ".env"
    if not cid and env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("MAL_CLIENT_ID") and "=" in line:
                cid = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not cid:
        raise RuntimeError(f"MAL_CLIENT_ID not found in env or {env_path}")
    return cid


class Client:
    """Paced GET with backoff on 429/5xx/network errors; one instance per host.

    get() returns the Response for terminal statuses (200/403/404/...) and None
    when retries are exhausted: the caller marks the unit as an error and moves
    on, so state stays resumable.
    """

    def __init__(self, min_interval: float, headers=None, tries: int = 6, timeout: int = 30):
        self.min_interval = min_interval
        self.tries = tries
        self.timeout = timeout
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        self._last = 0.0

    def _pace(self):
        wait = self._last + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def get(self, url, params=None):
        for attempt in range(self.tries):
            if STOP and attempt > 0:
                return None
            self._pace()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                print(f"    [-] {type(e).__name__} {url} (retry {attempt + 1}/{self.tries})")
                time.sleep(min(2 ** attempt, 60))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt * 2, 120))
                continue
            return r
        return None


SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    username        TEXT PRIMARY KEY COLLATE NOCASE,
    discovered_at   TEXT,
    ratings_status  TEXT,    -- NULL=pending | ok | empty | http_403 | http_404 | error
    ratings_at      TEXT,
    ratings_n       INTEGER,
    profile_status  TEXT,    -- NULL=pending | ok | http_404 | error
    profile_at      TEXT
);
"""


def db() -> sqlite3.Connection:
    RAW.mkdir(parents=True, exist_ok=True)
    # Three crawlers write this db concurrently; the default 5 s busy timeout can
    # raise "database is locked" on a slow disk and kill the run.
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

from .domain import Settings, digest, now, require_tgo_url, write_json


class SourceStore:
    """Immutable HTTP bytes, mutable check history, and per-field numeric lineage."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir
        for name in ("raw", "processed", "outputs", "models", "index"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY, request_key TEXT, url TEXT, method TEXT,
                    request_json TEXT, sha256 TEXT, path TEXT, retrieved_at TEXT,
                    content_type TEXT, last_modified TEXT, etag TEXT, title TEXT);
                CREATE INDEX IF NOT EXISTS source_request ON sources(request_key);
                CREATE TABLE IF NOT EXISTS checks (
                    id INTEGER PRIMARY KEY, request_key TEXT, checked_at TEXT,
                    source_id TEXT, status TEXT, changed INTEGER, error TEXT);
                CREATE TABLE IF NOT EXISTS facts (
                    run_id TEXT, entity TEXT, field TEXT, original TEXT,
                    numeric_value REAL, unit TEXT, source_id TEXT, locator TEXT,
                    PRIMARY KEY(run_id, entity, field, source_id, locator));
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, started_at TEXT, kind TEXT, metadata TEXT);
            """)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.root / "lineage.sqlite", timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def latest(self, request_key: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT s.* FROM checks c JOIN sources s ON s.source_id=c.source_id "
                             "WHERE c.request_key=? AND c.status='OK' ORDER BY c.id DESC LIMIT 1",
                             (request_key,)).fetchone()
        return dict(row) if row else None

    def fetch(self, url: str, *, data: dict | None = None, offline: bool = False,
              title: str = "") -> tuple[dict, bytes]:
        require_tgo_url(url)
        method = "POST" if data is not None else "GET"
        request_key = digest({"url": url, "method": method, "data": data})
        previous = self.latest(request_key)
        if offline:
            if previous is None:
                raise FileNotFoundError(f"No saved source for offline replay: {url}")
            raw = (self.root / previous["path"]).read_bytes()
            if digest(raw) != previous["sha256"]:
                raise ValueError(f"Saved source integrity check failed: {url}")
            return previous, raw
        failure = None
        for attempt in range(self.settings.retries + 1):
            try:
                with httpx.Client(timeout=self.settings.timeout_seconds,
                                  headers={"User-Agent": "TGOInsuranceResearch/0.1 (public data)"}) as client:
                    response = client.request(method, url, data=data)
                    for _ in range(5):
                        if not response.is_redirect:
                            break
                        target = require_tgo_url(urljoin(str(response.url), response.headers["location"]))
                        response = client.get(target)
                    response.raise_for_status()
                    raw = response.content
                    if not raw or len(raw) > 50 * 1024 * 1024:
                        raise ValueError("Empty or over-50-MiB source")
                    content_hash = digest(raw)
                    source_id = digest(request_key + content_hash)[:24]
                    suffix = ".pdf" if raw.startswith(b"%PDF") else ".csv" if ".csv" in url else ".json" if "application/json" in response.headers.get("content-type", "") else ".html"
                    relative = f"raw/{content_hash}{suffix}"
                    path = self.root / relative
                    if not path.exists():
                        with path.open("xb") as handle:
                            handle.write(raw)
                    elif digest(path.read_bytes()) != content_hash:
                        raise ValueError("Immutable raw-file hash collision or corruption")
                    source = {"source_id": source_id, "request_key": request_key, "url": url,
                              "method": method, "request_json": json.dumps(data, sort_keys=True),
                              "sha256": content_hash, "path": relative, "retrieved_at": now(),
                              "content_type": response.headers.get("content-type", ""),
                              "last_modified": response.headers.get("last-modified"),
                              "etag": response.headers.get("etag"), "title": title or url}
                    with self.connect() as db:
                        columns = ",".join(source)
                        db.execute(f"INSERT OR IGNORE INTO sources ({columns}) VALUES ({','.join('?' for _ in source)})", tuple(source.values()))
                        db.execute("INSERT INTO checks(request_key,checked_at,source_id,status,changed) VALUES(?,?,?,?,?)",
                                   (request_key, now(), source_id, "OK", int(previous is None or previous["sha256"] != content_hash)))
                        # Use the timestamp of the immutable version, not a fabricated new publication date.
                        source = dict(db.execute("SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone())
                    return source, raw
            except (httpx.HTTPError, ValueError) as exc:
                failure = exc
                if attempt < self.settings.retries:
                    time.sleep(min(2 ** attempt, 4))
        with self.connect() as db:
            db.execute("INSERT INTO checks(request_key,checked_at,status,error) VALUES(?,?,?,?)",
                       (request_key, now(), "FAILED", str(failure)[:500]))
        # Never silently substitute an old source for a failed fresh fetch.
        raise RuntimeError(f"TGO fetch failed for {url}: {failure}")

    def fact(self, run_id: str, entity: str, field: str, original: str,
             numeric_value: float | None, unit: str, source_id: str, locator: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO facts VALUES(?,?,?,?,?,?,?,?)",
                       (run_id, entity, field, original, numeric_value, unit, source_id, locator))

    def inventory(self, ids: list[str] | None = None) -> list[dict]:
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM sources ORDER BY url,source_id")]
        return [r for r in rows if ids is None or r["source_id"] in ids]

    def export(self, run_id: str, source_ids: list[str]) -> None:
        import pandas as pd
        pd.DataFrame(self.inventory(source_ids)).to_csv(self.root / "outputs/source_inventory.csv", index=False)
        with self.connect() as db:
            frame = pd.read_sql_query("SELECT * FROM facts WHERE run_id=?", db, params=(run_id,))
        frame.to_csv(self.root / "outputs/numeric_lineage.csv", index=False)
        write_json(self.root / "outputs/source_inventory.json", self.inventory(source_ids))

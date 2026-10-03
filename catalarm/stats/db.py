"""SQLite stats store (events, visits, performance samples). Stats only - no config or images."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    state TEXT NOT NULL,
    cat_id TEXT,
    confidence REAL,
    margin REAL,
    night INTEGER NOT NULL DEFAULT 0,
    snapshot TEXT
);
CREATE INDEX IF NOT EXISTS event_ts ON event(ts);
CREATE TABLE IF NOT EXISTS visit (
    id INTEGER PRIMARY KEY,
    camera_id TEXT NOT NULL,
    cat_id TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL
);
CREATE INDEX IF NOT EXISTS visit_start ON visit(start_ts);
CREATE TABLE IF NOT EXISTS perf_sample (
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    fps REAL,
    motion_ms REAL,
    detect_ms REAL,
    embed_ms REAL,
    analysed INTEGER
);
CREATE INDEX IF NOT EXISTS perf_ts ON perf_sample(ts);
"""


class StatsDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        # visits left open by a crash or restart are closed at their start time
        self._conn.execute("UPDATE visit SET end_ts = start_ts WHERE end_ts IS NULL")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, args)

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ---- writes --------------------------------------------------------------------------
    def add_event(
        self,
        camera_id: str,
        state: str,
        cat_id: str | None,
        confidence: float | None,
        margin: float | None,
        night: bool,
        snapshot: str | None,
        ts: float | None = None,
    ) -> int:
        cur = self._exec(
            "INSERT INTO event (ts, camera_id, state, cat_id, confidence, margin, night, snapshot)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts or time.time(), camera_id, state, cat_id, confidence, margin, int(night), snapshot),
        )
        return int(cur.lastrowid)

    def relabel_event(self, event_id: int, cat_id: str | None) -> None:
        self._exec(
            "UPDATE event SET cat_id = ?, state = COALESCE(?, 'no_cat') WHERE id = ?",
            (cat_id, cat_id, event_id),
        )

    def start_visit(self, camera_id: str, cat_id: str, ts: float) -> int:
        cur = self._exec(
            "INSERT INTO visit (camera_id, cat_id, start_ts) VALUES (?, ?, ?)", (camera_id, cat_id, ts)
        )
        return int(cur.lastrowid)

    def end_visit(self, visit_id: int, ts: float) -> None:
        self._exec("UPDATE visit SET end_ts = ? WHERE id = ?", (ts, visit_id))

    def add_perf(
        self,
        camera_id: str,
        fps: float,
        motion_ms: float,
        detect_ms: float,
        embed_ms: float,
        analysed: int,
        ts: float | None = None,
    ) -> None:
        self._exec(
            "INSERT INTO perf_sample VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts or time.time(), camera_id, fps, motion_ms, detect_ms, embed_ms, analysed),
        )

    def prune(self, older_than_days: int) -> list[str]:
        """Delete old rows, returns snapshot paths that are no longer referenced."""
        cutoff = time.time() - older_than_days * 86400
        snaps = [
            r["snapshot"]
            for r in self._rows("SELECT snapshot FROM event WHERE ts < ? AND snapshot IS NOT NULL", (cutoff,))
        ]
        self._exec("DELETE FROM event WHERE ts < ?", (cutoff,))
        self._exec("DELETE FROM visit WHERE start_ts < ?", (cutoff,))
        self._exec("DELETE FROM perf_sample WHERE ts < ?", (cutoff,))
        return snaps

    # ---- queries -------------------------------------------------------------------------
    def event(self, event_id: int) -> dict | None:
        rows = self._rows("SELECT * FROM event WHERE id = ?", (event_id,))
        return rows[0] if rows else None

    def events(
        self,
        camera_id: str | None = None,
        cat_id: str | None = None,
        state: str | None = None,
        before: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        sql, args = "SELECT * FROM event WHERE 1=1", []
        for col, val in (("camera_id", camera_id), ("cat_id", cat_id), ("state", state)):
            if val:
                sql += f" AND {col} = ?"
                args.append(val)
        if before:
            sql += " AND ts < ?"
            args.append(before)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return self._rows(sql, tuple(args))

    def daily_visits(self, days: int, tz_offset_s: int = 0) -> list[dict]:
        since = time.time() - days * 86400
        return self._rows(
            "SELECT date(start_ts + ?, 'unixepoch') AS day, cat_id, camera_id, COUNT(*) AS visits,"
            " SUM(COALESCE(end_ts, start_ts) - start_ts) AS seconds"
            " FROM visit WHERE start_ts >= ? GROUP BY day, cat_id, camera_id ORDER BY day",
            (tz_offset_s, since),
        )

    def hourly_visits(self, days: int, tz_offset_s: int = 0) -> list[dict]:
        since = time.time() - days * 86400
        return self._rows(
            "SELECT CAST(strftime('%H', start_ts + ?, 'unixepoch') AS INTEGER) AS hour, cat_id,"
            " COUNT(*) AS visits FROM visit WHERE start_ts >= ? GROUP BY hour, cat_id",
            (tz_offset_s, since),
        )

    def state_counts(self, days: int) -> list[dict]:
        since = time.time() - days * 86400
        return self._rows(
            "SELECT camera_id, state, COUNT(*) AS n FROM event WHERE ts >= ? GROUP BY camera_id, state",
            (since,),
        )

    def perf(self, hours: int, camera_id: str | None = None) -> list[dict]:
        since = time.time() - hours * 3600
        sql = (
            "SELECT CAST(ts / 600 AS INTEGER) * 600 AS ts, camera_id, AVG(fps) AS fps,"
            " AVG(motion_ms) AS motion_ms, AVG(detect_ms) AS detect_ms, AVG(embed_ms) AS embed_ms,"
            " SUM(analysed) AS analysed FROM perf_sample WHERE ts >= ?"
        )
        args: list = [since]
        if camera_id:
            sql += " AND camera_id = ?"
            args.append(camera_id)
        return self._rows(sql + " GROUP BY 1, camera_id ORDER BY 1", tuple(args))

"""Tiny SQLite layer: durable analyst audit trail + feedback (survives restarts). ':memory:' works for tests."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, ts TEXT NOT NULL,
    action TEXT NOT NULL, actor TEXT NOT NULL, status TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, txn_id TEXT NOT NULL, verdict TEXT NOT NULL,
    analyst TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_audit_case ON audit(case_id);
CREATE INDEX IF NOT EXISTS ix_feedback_case ON feedback(case_id);
"""


class Db:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, args).fetchall())

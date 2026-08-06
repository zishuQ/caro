"""Caro task lifecycle persistence in SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from caro.errors import RunNotFoundError


class RunStatus(StrEnum):
    """Durable task lifecycle states."""

    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.WAITING_APPROVAL, RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.WAITING_APPROVAL: frozenset(
        {RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.PAUSED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One persisted Caro task."""

    run_id: str
    thread_id: str
    task: str
    workspace: str
    status: RunStatus
    config_hash: str
    created_at: str
    updated_at: str
    parent_run_id: str | None = None
    checkpoint_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Append-only lifecycle audit event."""

    event_id: int
    run_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        thread_id=row["thread_id"],
        task=row["task"],
        workspace=row["workspace"],
        status=RunStatus(row["status"]),
        config_hash=row["config_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        parent_run_id=row["parent_run_id"],
        checkpoint_id=row["checkpoint_id"],
        error=row["error"],
    )


class RunStore:
    """Small SQLite repository independent of LangGraph's checkpoint schema."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    parent_run_id TEXT REFERENCES runs(run_id),
                    checkpoint_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_thread_id ON runs(thread_id);
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id, event_id);
                """
            )

    def create(
        self,
        *,
        task: str,
        workspace: Path,
        config_hash: str,
        run_id: str | None = None,
        thread_id: str | None = None,
        parent_run_id: str | None = None,
        checkpoint_id: str | None = None,
        status: RunStatus = RunStatus.CREATED,
    ) -> RunRecord:
        """Create a run and its first audit event."""

        created_at = _now()
        resolved_run_id = run_id or str(uuid.uuid4())
        resolved_thread_id = thread_id or str(uuid.uuid4())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, thread_id, task, workspace, status, config_hash,
                    parent_run_id, checkpoint_id, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    resolved_run_id,
                    resolved_thread_id,
                    task,
                    str(workspace.expanduser().resolve()),
                    status.value,
                    config_hash,
                    parent_run_id,
                    checkpoint_id,
                    created_at,
                    created_at,
                ),
            )
            connection.execute(
                "INSERT INTO run_events (run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (resolved_run_id, "created", "{}", created_at),
            )
        return self.get(resolved_run_id)

    def get(self, run_id: str) -> RunRecord:
        """Return a run or raise a domain-specific not-found error."""

        with self._connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return _to_record(row)

    def list_runs(self, *, limit: int = 50, status: RunStatus | None = None) -> list[RunRecord]:
        """List recent runs, newest first."""

        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connection() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
        return [_to_record(row) for row in rows]

    def transition(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        """Validate and persist a lifecycle transition."""

        timestamp = _now()
        event_payload = dict(payload or {})
        if error is not None:
            event_payload["error"] = error
        with self._connection() as connection:
            # Serialize lifecycle transitions so cancellation cannot be lost
            # between a read and the subsequent terminal update.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            current = _to_record(row)
            if status == current.status:
                return current
            if status not in _ALLOWED_TRANSITIONS[current.status]:
                raise ValueError(f"invalid run transition: {current.status.value} -> {status.value}")
            updated = connection.execute(
                "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
                (status.value, error, timestamp, run_id),
            )
            if updated.rowcount != 1:
                raise ValueError(
                    f"run transition raced with another lifecycle update: {run_id}"
                )
            connection.execute(
                "INSERT INTO run_events (run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (run_id, status.value, json.dumps(event_payload, sort_keys=True), timestamp),
            )
            result = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if result is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            return _to_record(result)

    def set_checkpoint(self, run_id: str, checkpoint_id: str | None) -> RunRecord:
        """Persist the latest known LangGraph checkpoint pointer."""

        self.get(run_id)
        timestamp = _now()
        with self._connection() as connection:
            connection.execute(
                "UPDATE runs SET checkpoint_id = ?, updated_at = ? WHERE run_id = ?",
                (checkpoint_id, timestamp, run_id),
            )
            connection.execute(
                "INSERT INTO run_events (run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    run_id,
                    "checkpoint",
                    json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True),
                    timestamp,
                ),
            )
        return self.get(run_id)

    def events(self, run_id: str) -> list[RunEvent]:
        """Return append-only lifecycle events for one run."""

        self.get(run_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return [
            RunEvent(
                event_id=row["event_id"],
                run_id=row["run_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def fork(self, parent_run_id: str, *, checkpoint_id: str, config_hash: str) -> RunRecord:
        """Create a paused child run sharing the parent LangGraph thread branch."""

        parent = self.get(parent_run_id)
        return self.create(
            task=parent.task,
            workspace=Path(parent.workspace),
            config_hash=config_hash,
            thread_id=parent.thread_id,
            parent_run_id=parent.run_id,
            checkpoint_id=checkpoint_id,
            status=RunStatus.PAUSED,
        )

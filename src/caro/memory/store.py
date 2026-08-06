"""SQLite and FTS5 repository for governed structured memory."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from caro.memory.models import (
    MemoryAuthorization,
    MemoryEvent,
    MemoryItem,
    MemoryKind,
    MemoryOperation,
    MemoryPutResult,
    MemorySearchResult,
    MemoryStats,
    MemoryStatus,
)

_SCOPE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?::[A-Za-z0-9][A-Za-z0-9_.@/-]{0,127})?$")
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_QUERY_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
    r"(\s*[:=]\s*)[^\s,;]+"
)
_QUERY_TOKEN = re.compile(r"(?i)\b(?:sk|ghp|github_pat|xox[baprs])-[-A-Za-z0-9_]{12,}\b")
_QUERY_AUTHORIZATION = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}\b")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("memory timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else _as_utc(value).isoformat()


def _datetime(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _normalize_content(content: str) -> str:
    normalized = " ".join(content.split())
    if not normalized:
        raise ValueError("memory content cannot be empty")
    if len(normalized) > 20_000:
        raise ValueError("memory content cannot exceed 20000 characters")
    return normalized


def _validate_scope(scope: str) -> str:
    if _SCOPE_PATTERN.fullmatch(scope) is None:
        raise ValueError(f"invalid memory scope: {scope!r}")
    return scope


def _fts_query(query: str) -> str:
    tokens = _TOKEN_PATTERN.findall(query)[:16]
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _safe_audit_query(query: str) -> str:
    """Keep bounded, redacted query context out of the memory audit log."""

    bounded = query[:512]
    bounded = _QUERY_SECRET_ASSIGNMENT.sub(r"\1=[REDACTED]", bounded)
    bounded = _QUERY_TOKEN.sub("[REDACTED]", bounded)
    return _QUERY_AUTHORIZATION.sub(lambda match: f"{match.group(0).split()[0]} [REDACTED]", bounded)


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    metadata = json.loads(row["metadata_json"])
    if not isinstance(metadata, dict):
        raise TypeError("memory metadata must decode to a mapping")
    return MemoryItem(
        memory_id=str(row["memory_id"]),
        kind=MemoryKind(row["kind"]),
        scope=str(row["scope"]),
        content=str(row["content"]),
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        source_run_id=row["source_run_id"],
        source_agent=row["source_agent"],
        status=MemoryStatus(row["status"]),
        content_hash=str(row["content_hash"]),
        metadata=dict(metadata),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_accessed_at=_datetime(row["last_accessed_at"]),
        expires_at=_datetime(row["expires_at"]),
        access_count=int(row["access_count"]),
    )


class MemoryStore:
    """Transactional structured-memory repository isolated to one workspace database."""

    def __init__(self, path: Path, *, allowed_root: Path | None = None) -> None:
        raw_path = path.expanduser().absolute()
        if raw_path.is_symlink():
            raise ValueError(f"structured-memory database cannot be a symbolic link: {raw_path}")
        resolved_path = raw_path.resolve()
        if allowed_root is not None:
            root = allowed_root.expanduser().resolve()
            try:
                resolved_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"structured-memory database resolves outside the allowed root: {resolved_path}"
                ) from error
        self.path = resolved_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_tokenizer = "unknown"
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
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('episodic', 'semantic', 'procedural')),
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL CHECK (importance >= 0 AND importance <= 1),
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    source_run_id TEXT,
                    source_agent TEXT,
                    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_active_identity
                    ON memory_items(scope, kind, content_hash) WHERE status = 'active';
                CREATE INDEX IF NOT EXISTS idx_memory_scope_status
                    ON memory_items(scope, status, importance DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_expiry ON memory_items(expires_at);
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    actor TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_item
                    ON memory_events(memory_id, event_id);
                CREATE TABLE IF NOT EXISTS memory_operations (
                    operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    authorization TEXT NOT NULL,
                    source_run_id TEXT,
                    source_agent TEXT,
                    memory_id TEXT REFERENCES memory_items(memory_id),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_operations_run
                    ON memory_operations(source_run_id, operation_id DESC);
                """
            )
            self._ensure_fts(connection)

    def _ensure_fts(self, connection: sqlite3.Connection) -> None:
        """Prefer trigram substring retrieval and migrate an older unicode61 index."""

        existing = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
        ).fetchone()
        if existing is not None and "trigram" in str(existing["sql"]).casefold():
            self._fts_tokenizer = "trigram"
            return
        if existing is not None:
            connection.execute("DROP TABLE memory_fts")
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    tokenize = 'trigram'
                )
                """
            )
            self._fts_tokenizer = "trigram"
        except sqlite3.OperationalError:
            connection.execute(
                """
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                )
                """
            )
            self._fts_tokenizer = "unicode61"
        connection.execute(
            """
            INSERT INTO memory_fts(memory_id, content)
            SELECT memory_id, content FROM memory_items WHERE status = 'active'
            """
        )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        memory_id: str,
        event_type: str,
        *,
        actor: str | None = None,
        payload: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_events(memory_id, event_type, actor, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                event_type,
                actor,
                json.dumps(payload or {}, sort_keys=True, ensure_ascii=False),
                _iso(created_at or _utc_now()),
            ),
        )

    @staticmethod
    def _record_operation_row(
        connection: sqlite3.Connection,
        *,
        content_hash: str,
        kind: MemoryKind,
        scope: str,
        outcome: str,
        reason: str,
        authorization: MemoryAuthorization,
        source_run_id: str | None,
        source_agent: str | None,
        memory_id: str | None,
        created_at: datetime,
    ) -> int:
        """Insert one operation row on the caller's active transaction."""

        cursor = connection.execute(
            """
            INSERT INTO memory_operations(
                content_hash, kind, scope, outcome, reason, authorization,
                source_run_id, source_agent, memory_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_hash,
                kind.value,
                scope,
                outcome,
                reason,
                authorization.value,
                source_run_id,
                source_agent,
                memory_id,
                _iso(created_at),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite returned no operation ID")
        return int(cursor.lastrowid)

    def put(
        self,
        *,
        kind: MemoryKind,
        scope: str,
        content: str,
        importance: float,
        confidence: float,
        source_run_id: str | None = None,
        source_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        supersedes_id: str | None = None,
        actor: str | None = None,
        operation_authorization: MemoryAuthorization | None = None,
    ) -> MemoryPutResult:
        """Create one active item, deduplicate exact content, or atomically supersede an item."""

        normalized = _normalize_content(content)
        validated_scope = _validate_scope(scope)
        if not 0 <= importance <= 1:
            raise ValueError("memory importance must be between 0 and 1")
        if not 0 <= confidence <= 1:
            raise ValueError("memory confidence must be between 0 and 1")
        expiry = _iso(expires_at)
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        now = _utc_now()
        memory_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False)

        with self._connection() as connection:
            duplicate = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE scope = ? AND kind = ? AND content_hash = ? AND status = 'active'
                """,
                (validated_scope, kind.value, content_hash),
            ).fetchone()
            if duplicate is not None:
                item = _row_to_item(duplicate)
                self._event(
                    connection,
                    item.memory_id,
                    "duplicate_ignored",
                    actor=actor or source_agent,
                    payload={"source_run_id": source_run_id},
                    created_at=now,
                )
                if operation_authorization is not None:
                    self._record_operation_row(
                        connection,
                        content_hash=content_hash,
                        kind=kind,
                        scope=validated_scope,
                        outcome="duplicate",
                        reason="exact active memory already exists",
                        authorization=operation_authorization,
                        source_run_id=source_run_id,
                        source_agent=source_agent,
                        memory_id=item.memory_id,
                        created_at=now,
                    )
                return MemoryPutResult(item=item, created=False)

            superseded: sqlite3.Row | None = None
            if supersedes_id is not None:
                superseded = connection.execute(
                    "SELECT * FROM memory_items WHERE memory_id = ? AND status = 'active'",
                    (supersedes_id,),
                ).fetchone()
                if superseded is None:
                    raise ValueError(f"active superseded memory not found: {supersedes_id}")
                if superseded["scope"] != validated_scope or superseded["kind"] != kind.value:
                    raise ValueError("superseded memory must have the same kind and scope")

            try:
                connection.execute(
                    """
                    INSERT INTO memory_items(
                        memory_id, kind, scope, content, importance, confidence,
                        source_run_id, source_agent, status, content_hash, metadata_json,
                        created_at, updated_at, last_accessed_at, expires_at, access_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, NULL, ?, 0)
                    """,
                    (
                        memory_id,
                        kind.value,
                        validated_scope,
                        normalized,
                        importance,
                        confidence,
                        source_run_id,
                        source_agent,
                        content_hash,
                        metadata_json,
                        _iso(now),
                        _iso(now),
                        expiry,
                    ),
                )
            except sqlite3.IntegrityError:
                connection.rollback()
                duplicate = connection.execute(
                    """
                    SELECT * FROM memory_items
                    WHERE scope = ? AND kind = ? AND content_hash = ? AND status = 'active'
                    """,
                    (validated_scope, kind.value, content_hash),
                ).fetchone()
                if duplicate is None:
                    raise
                item = _row_to_item(duplicate)
                self._event(
                    connection,
                    item.memory_id,
                    "duplicate_ignored",
                    actor=actor or source_agent,
                    payload={"source_run_id": source_run_id},
                    created_at=now,
                )
                if operation_authorization is not None:
                    self._record_operation_row(
                        connection,
                        content_hash=content_hash,
                        kind=kind,
                        scope=validated_scope,
                        outcome="duplicate",
                        reason="exact active memory already exists",
                        authorization=operation_authorization,
                        source_run_id=source_run_id,
                        source_agent=source_agent,
                        memory_id=item.memory_id,
                        created_at=now,
                    )
                return MemoryPutResult(item=item, created=False)
            connection.execute(
                "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
                (memory_id, normalized),
            )
            self._event(
                connection,
                memory_id,
                "created",
                actor=actor or source_agent,
                payload={"source_run_id": source_run_id, "supersedes_id": supersedes_id},
                created_at=now,
            )
            if superseded is not None and supersedes_id is not None:
                connection.execute(
                    "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                    (_iso(now), supersedes_id),
                )
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (supersedes_id,))
                self._event(
                    connection,
                    supersedes_id,
                    "superseded",
                    actor=actor or source_agent,
                    payload={"replacement_id": memory_id},
                    created_at=now,
                )
            if operation_authorization is not None:
                self._record_operation_row(
                    connection,
                    content_hash=content_hash,
                    kind=kind,
                    scope=validated_scope,
                    outcome="created",
                    reason="memory committed",
                    authorization=operation_authorization,
                    source_run_id=source_run_id,
                    source_agent=source_agent,
                    memory_id=memory_id,
                    created_at=now,
                )
            row = connection.execute("SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,)).fetchone()
            if row is None:
                raise RuntimeError("new memory row disappeared before commit")
            return MemoryPutResult(item=_row_to_item(row), created=True)

    def get(self, memory_id: str, *, include_archived: bool = False) -> MemoryItem:
        """Return an item by ID."""

        query = "SELECT * FROM memory_items WHERE memory_id = ?"
        parameters: tuple[Any, ...] = (memory_id,)
        if not include_archived:
            query += " AND status = 'active'"
        with self._connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError(f"memory not found: {memory_id}")
        return _row_to_item(row)

    def search(
        self,
        query: str,
        *,
        scopes: Sequence[str],
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 8,
        min_importance: float = 0.0,
        now: datetime | None = None,
        record_access: bool = True,
    ) -> list[MemorySearchResult]:
        """Retrieve active, unexpired memories with FTS5 and metadata filters."""

        if not scopes:
            raise ValueError("at least one memory scope is required")
        validated_scopes = tuple(_validate_scope(scope) for scope in scopes)
        if not 1 <= limit <= 50:
            raise ValueError("memory search limit must be between 1 and 50")
        if not 0 <= min_importance <= 1:
            raise ValueError("minimum importance must be between 0 and 1")
        current = _as_utc(now or _utc_now())
        fts = _fts_query(query)
        if not fts:
            return []
        conditions = [
            "m.status = 'active'",
            f"m.scope IN ({','.join('?' for _ in validated_scopes)})",
            "m.importance >= ?",
            "(m.expires_at IS NULL OR m.expires_at > ?)",
        ]
        filter_parameters: list[Any] = [*validated_scopes, min_importance, _iso(current)]
        if kinds:
            conditions.append(f"m.kind IN ({','.join('?' for _ in kinds)})")
            filter_parameters.extend(kind.value for kind in kinds)

        if fts:
            sql = (
                "SELECT m.*, bm25(memory_fts) AS fts_rank "
                "FROM memory_fts JOIN memory_items AS m ON m.memory_id = memory_fts.memory_id "
                "WHERE memory_fts MATCH ? AND "
                + " AND ".join(conditions)
                + " ORDER BY fts_rank ASC, m.importance DESC, m.confidence DESC, m.updated_at DESC LIMIT ?"
            )
            parameters = [fts, *filter_parameters, limit]
        else:
            sql = (
                "SELECT m.*, NULL AS fts_rank FROM memory_items AS m WHERE "
                + " AND ".join(conditions)
                + " ORDER BY m.importance DESC, m.confidence DESC, m.updated_at DESC LIMIT ?"
            )
            parameters = [*filter_parameters, limit]

        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            if fts and not rows:
                tokens = _TOKEN_PATTERN.findall(query)[:16]
                like_clause = " OR ".join("m.content LIKE ? ESCAPE '\\'" for _ in tokens)
                like_patterns = [
                    "%" + token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                    for token in tokens
                ]
                fallback_sql = (
                    "SELECT m.*, NULL AS fts_rank FROM memory_items AS m WHERE ("
                    + like_clause
                    + ") AND "
                    + " AND ".join(conditions)
                    + " ORDER BY m.importance DESC, m.confidence DESC, m.updated_at DESC LIMIT ?"
                )
                rows = connection.execute(
                    fallback_sql,
                    [*like_patterns, *filter_parameters, limit],
                ).fetchall()
            results: list[MemorySearchResult] = []
            for index, row in enumerate(rows):
                item = _row_to_item(row)
                relevance = 1.0 / (1 + index) if fts else 0.5
                score = min(1.0, relevance * 0.6 + item.importance * 0.25 + item.confidence * 0.15)
                if record_access:
                    connection.execute(
                        """
                        UPDATE memory_items
                        SET access_count = access_count + 1, last_accessed_at = ?, updated_at = updated_at
                        WHERE memory_id = ?
                        """,
                        (_iso(current), item.memory_id),
                    )
                    self._event(
                        connection,
                        item.memory_id,
                        "retrieved",
                        payload={"query": _safe_audit_query(query), "rank": index + 1},
                        created_at=current,
                    )
                    item = replace(item, access_count=item.access_count + 1, last_accessed_at=current)
                results.append(MemorySearchResult(item=item, score=score, rank=index + 1))
            return results

    def archive(
        self,
        memory_id: str,
        *,
        allowed_scopes: Sequence[str],
        reason: str,
        actor: str | None,
        source_run_id: str | None = None,
    ) -> MemoryItem:
        """Soft-delete one item after enforcing the caller's scope boundary."""

        scopes = {_validate_scope(scope) for scope in allowed_scopes}
        if not reason.strip():
            raise ValueError("archive reason cannot be empty")
        now = _utc_now()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_items WHERE memory_id = ? AND status = 'active'",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"active memory not found: {memory_id}")
            if row["scope"] not in scopes:
                raise PermissionError(f"memory scope is not accessible: {row['scope']}")
            connection.execute(
                "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                (_iso(now), memory_id),
            )
            connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            self._event(
                connection,
                memory_id,
                "archived",
                actor=actor,
                payload={"reason": _safe_audit_query(reason.strip()), "source_run_id": source_run_id},
                created_at=now,
            )
            updated = connection.execute("SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,)).fetchone()
            if updated is None:
                raise RuntimeError("archived memory row disappeared before commit")
            return _row_to_item(updated)

    def prune_expired(self, *, now: datetime | None = None, actor: str = "caro-retention") -> int:
        """Archive expired items without physically deleting their audit history."""

        current = _as_utc(now or _utc_now())
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT memory_id FROM memory_items WHERE status = 'active' AND expires_at <= ?",
                (_iso(current),),
            ).fetchall()
            for row in rows:
                memory_id = str(row["memory_id"])
                connection.execute(
                    "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                    (_iso(current), memory_id),
                )
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
                self._event(
                    connection,
                    memory_id,
                    "expired",
                    actor=actor,
                    created_at=current,
                )
            return len(rows)

    def events(self, memory_id: str) -> list[MemoryEvent]:
        """Return append-only events for one memory item."""

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY event_id",
                (memory_id,),
            ).fetchall()
        return [
            MemoryEvent(
                event_id=int(row["event_id"]),
                memory_id=str(row["memory_id"]),
                event_type=str(row["event_type"]),
                actor=row["actor"],
                payload=dict(json.loads(row["payload_json"])),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def stats(self) -> MemoryStats:
        """Return active/archive totals and active counts by kind and scope."""

        with self._connection() as connection:
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM memory_items GROUP BY status"
            ).fetchall()
            kind_rows = connection.execute(
                "SELECT kind, COUNT(*) AS count FROM memory_items WHERE status = 'active' GROUP BY kind"
            ).fetchall()
            scope_rows = connection.execute(
                "SELECT scope, COUNT(*) AS count FROM memory_items WHERE status = 'active' GROUP BY scope"
            ).fetchall()
        statuses = {str(row["status"]): int(row["count"]) for row in status_rows}
        return MemoryStats(
            active=statuses.get(MemoryStatus.ACTIVE.value, 0),
            archived=statuses.get(MemoryStatus.ARCHIVED.value, 0),
            by_kind={MemoryKind(row["kind"]): int(row["count"]) for row in kind_rows},
            by_scope={str(row["scope"]): int(row["count"]) for row in scope_rows},
        )

    def record_operation(
        self,
        *,
        content_hash: str,
        kind: MemoryKind,
        scope: str,
        outcome: str,
        reason: str,
        authorization: MemoryAuthorization,
        source_run_id: str | None,
        source_agent: str | None,
        memory_id: str | None,
        created_at: datetime | None = None,
    ) -> MemoryOperation:
        """Audit a consolidation without storing rejected candidate content."""

        _validate_scope(scope)
        timestamp = _as_utc(created_at or _utc_now())
        with self._connection() as connection:
            operation_id = self._record_operation_row(
                connection,
                content_hash=content_hash,
                kind=kind,
                scope=scope,
                outcome=outcome,
                reason=reason,
                authorization=authorization,
                source_run_id=source_run_id,
                source_agent=source_agent,
                memory_id=memory_id,
                created_at=timestamp,
            )
        return MemoryOperation(
            operation_id=operation_id,
            content_hash=content_hash,
            kind=kind,
            scope=scope,
            outcome=outcome,
            reason=reason,
            authorization=authorization,
            source_run_id=source_run_id,
            source_agent=source_agent,
            memory_id=memory_id,
            created_at=timestamp,
        )

    def operations(self, *, limit: int = 50, source_run_id: str | None = None) -> list[MemoryOperation]:
        """Return recent content-free consolidation audit records."""

        if not 1 <= limit <= 1000:
            raise ValueError("operation limit must be between 1 and 1000")
        with self._connection() as connection:
            if source_run_id is None:
                rows = connection.execute(
                    "SELECT * FROM memory_operations ORDER BY operation_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM memory_operations
                    WHERE source_run_id = ? ORDER BY operation_id DESC LIMIT ?
                    """,
                    (source_run_id, limit),
                ).fetchall()
        return [
            MemoryOperation(
                operation_id=int(row["operation_id"]),
                content_hash=str(row["content_hash"]),
                kind=MemoryKind(row["kind"]),
                scope=str(row["scope"]),
                outcome=str(row["outcome"]),
                reason=str(row["reason"]),
                authorization=MemoryAuthorization(row["authorization"]),
                source_run_id=row["source_run_id"],
                source_agent=row["source_agent"],
                memory_id=row["memory_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

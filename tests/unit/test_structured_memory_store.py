"""SQLite structured-memory repository tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from caro.memory.models import MemoryKind, MemoryPutResult, MemoryStatus
from caro.memory.store import MemoryStore


def test_store_creates_searches_and_audits_structured_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")

    result = store.put(
        kind=MemoryKind.PROCEDURAL,
        scope="project",
        content="Run uv run pytest -q before committing parser changes.",
        importance=0.8,
        confidence=0.95,
        source_run_id="run-1",
        source_agent="tester",
        metadata={"evidence": "tests passed"},
    )

    assert result.created is True
    assert result.item.source_run_id == "run-1"
    assert result.item.metadata == {"evidence": "tests passed"}
    matches = store.search("pytest parser", scopes=("project",), limit=5)
    assert [match.item.memory_id for match in matches] == [result.item.memory_id]
    assert matches[0].score > 0
    assert store.get(result.item.memory_id).access_count == 1
    assert [event.event_type for event in store.events(result.item.memory_id)] == ["created", "retrieved"]


def test_exact_active_memory_is_idempotent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The public API lives in src/caro/api.py.",
        importance=0.7,
        confidence=0.9,
    )
    second = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="  The public API lives in src/caro/api.py.  ",
        importance=0.9,
        confidence=1.0,
    )

    assert first.created is True
    assert second.created is False
    assert second.item.memory_id == first.item.memory_id
    assert store.stats().active == 1


def test_concurrent_exact_memory_puts_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first_store = MemoryStore(path)
    second_store = MemoryStore(path)

    def put(store: MemoryStore) -> MemoryPutResult:
        return store.put(
            kind=MemoryKind.SEMANTIC,
            scope="project",
            content="The parser entry point is src/parser.py.",
            importance=0.8,
            confidence=1.0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(put, (first_store, second_store)))
    assert sum(result.created for result in results) == 1
    assert results[0].item.memory_id == results[1].item.memory_id


def test_search_enforces_scope_kind_importance_and_expiry(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    now = datetime.now(UTC)
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="role:reviewer",
        content="Parser regression reproduced on Windows.",
        importance=0.8,
        confidence=0.8,
    )
    store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="Parser supports Windows paths.",
        importance=0.4,
        confidence=0.9,
    )
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="project",
        content="Old parser experiment on Windows.",
        importance=0.9,
        confidence=0.9,
        expires_at=now - timedelta(seconds=1),
    )

    project = store.search("parser Windows", scopes=("project",), min_importance=0.5, now=now)
    reviewer = store.search(
        "parser Windows",
        scopes=("project", "role:reviewer"),
        kinds=(MemoryKind.EPISODIC,),
        now=now,
    )

    assert project == []
    assert [match.item.scope for match in reviewer] == ["role:reviewer"]


def test_archive_is_soft_delete_and_scope_guarded(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    created = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="role:architect",
        content="The package has no circular imports.",
        importance=0.6,
        confidence=0.7,
    ).item

    with pytest.raises(PermissionError, match="scope"):
        store.archive(created.memory_id, allowed_scopes=("project",), reason="wrong", actor="reviewer")

    archived = store.archive(
        created.memory_id,
        allowed_scopes=("role:architect",),
        reason="superseded by evidence",
        actor="architect",
    )

    assert archived.status is MemoryStatus.ARCHIVED
    assert store.search("circular imports", scopes=("role:architect",)) == []
    assert store.get(created.memory_id, include_archived=True).status is MemoryStatus.ARCHIVED
    assert store.events(created.memory_id)[-1].payload["reason"] == "superseded by evidence"


def test_supersede_archives_old_item_and_commits_new_item_atomically(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    old = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The API supports Python 3.10.",
        importance=0.7,
        confidence=0.8,
    ).item

    replacement = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The API supports Python 3.11 and newer.",
        importance=0.9,
        confidence=1.0,
        supersedes_id=old.memory_id,
    )

    assert replacement.created is True
    assert replacement.item.memory_id != old.memory_id
    assert store.get(old.memory_id, include_archived=True).status is MemoryStatus.ARCHIVED
    assert store.search("Python API", scopes=("project",))[0].item.memory_id == replacement.item.memory_id

    with pytest.raises(ValueError, match="same kind and scope"):
        store.put(
            kind=MemoryKind.PROCEDURAL,
            scope="project",
            content="Use Python 3.11.",
            importance=0.9,
            confidence=1.0,
            supersedes_id=replacement.item.memory_id,
        )

    assert store.get(replacement.item.memory_id).status is MemoryStatus.ACTIVE


def test_store_rejects_database_symlink_or_parent_escape_when_workspace_bound(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    outside_database = outside / "memory.sqlite3"
    outside_database.touch()
    direct_link = workspace / "memory.sqlite3"
    direct_link.symlink_to(outside_database)

    with pytest.raises(ValueError, match="symbolic link"):
        MemoryStore(direct_link, allowed_root=workspace)

    state_link = workspace / ".caro"
    state_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the allowed root"):
        MemoryStore(state_link / "memory.sqlite3", allowed_root=workspace)


def test_search_supports_cjk_substring_retrieval(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    created = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="解析器入口位于 src/parser.py, 并负责加载配置。",
        importance=0.9,
        confidence=1,
    ).item

    results = store.search("解析器入口", scopes=("project",))

    assert [result.item.memory_id for result in results] == [created.memory_id]

"""SQLite RunStore tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from caro.errors import RunNotFoundError
from caro.state.runs import RunStatus, RunStore


def test_run_lifecycle_and_events_persist(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    run = store.create(task="fix parser", workspace=tmp_path, config_hash="abc")
    assert run.status is RunStatus.CREATED

    store.transition(run.run_id, RunStatus.RUNNING)
    store.set_checkpoint(run.run_id, "cp-1")
    completed = store.transition(run.run_id, RunStatus.COMPLETED)

    assert completed.status is RunStatus.COMPLETED
    assert completed.checkpoint_id == "cp-1"
    assert [event.event_type for event in store.events(run.run_id)] == [
        "created",
        "running",
        "checkpoint",
        "completed",
    ]


def test_invalid_transition_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    run = store.create(task="task", workspace=tmp_path, config_hash="abc")
    with pytest.raises(ValueError, match="invalid run transition"):
        store.transition(run.run_id, RunStatus.COMPLETED)


def test_failed_run_can_resume(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    run = store.create(task="task", workspace=tmp_path, config_hash="abc")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED, error="network")
    resumed = store.transition(run.run_id, RunStatus.RUNNING)
    assert resumed.error is None


def test_fork_keeps_thread_and_links_parent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    parent = store.create(task="task", workspace=tmp_path, config_hash="old")
    child = store.fork(parent.run_id, checkpoint_id="fork-cp", config_hash="new")
    assert child.thread_id == parent.thread_id
    assert child.parent_run_id == parent.run_id
    assert child.checkpoint_id == "fork-cp"
    assert child.status is RunStatus.PAUSED


def test_get_unknown_run_raises_domain_error(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError):
        RunStore(tmp_path / "runs.sqlite3").get("missing")


def test_non_terminal_run_can_be_cancelled_with_audit_payload(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    run = store.create(task="task", workspace=tmp_path, config_hash="abc")

    cancelled = store.transition(
        run.run_id,
        RunStatus.CANCELLED,
        payload={"reason": "operator requested cancellation"},
    )

    assert cancelled.status is RunStatus.CANCELLED
    assert store.events(run.run_id)[-1].payload["reason"] == "operator requested cancellation"


def test_concurrent_terminal_transitions_have_one_winner(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    run = store.create(task="task", workspace=tmp_path, config_hash="abc")
    store.transition(run.run_id, RunStatus.RUNNING)

    def transition(status: RunStatus) -> str:
        try:
            store.transition(run.run_id, status)
        except ValueError:
            return "rejected"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                transition,
                (RunStatus.COMPLETED, RunStatus.CANCELLED),
            )
        )
    assert sorted(outcomes) == ["accepted", "rejected"]
    assert store.get(run.run_id).status in {RunStatus.COMPLETED, RunStatus.CANCELLED}

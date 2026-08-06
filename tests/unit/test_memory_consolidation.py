"""Governed LangGraph memory-consolidation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from caro.memory.graph import MemoryConsolidator
from caro.memory.models import (
    ConsolidationOutcome,
    MemoryAuthorization,
    MemoryKind,
    MemoryProposal,
)
from caro.memory.policy import MemoryPolicy
from caro.memory.store import MemoryStore


def build_consolidator(tmp_path: Path) -> tuple[MemoryStore, MemoryConsolidator]:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    policy = MemoryPolicy(minimum_confidence=0.6, max_content_chars=4000, episodic_ttl_days=90)
    return store, MemoryConsolidator(store, policy)


def test_verified_memory_flows_through_named_langgraph_nodes(tmp_path: Path) -> None:
    store, consolidator = build_consolidator(tmp_path)
    proposal = MemoryProposal(
        kind=MemoryKind.PROCEDURAL,
        scope="project",
        content="Run uv run pytest -q before committing changes.",
        importance=0.8,
        confidence=0.95,
        evidence="The command completed successfully in run-1.",
        source_run_id="run-1",
        source_agent="tester",
    )

    result = consolidator.consolidate(proposal, authorization=MemoryAuthorization.GOVERNED_TOOL)

    assert result.outcome is ConsolidationOutcome.CREATED
    assert result.item is not None
    assert store.get(result.item.memory_id).content == proposal.content
    graph_nodes = set(consolidator.graph.get_graph().nodes)
    assert {"encode", "policy", "retrieve_existing", "commit"}.issubset(graph_nodes)
    operation = store.operations(limit=1)[0]
    assert operation.outcome == ConsolidationOutcome.CREATED.value
    assert operation.source_run_id == "run-1"


def test_secret_and_personal_data_are_rejected_before_persistence(tmp_path: Path) -> None:
    store, consolidator = build_consolidator(tmp_path)
    proposals = [
        MemoryProposal(
            kind=MemoryKind.SEMANTIC,
            scope="project",
            content="Production API_KEY=sk-super-secret-value-123456.",
            importance=1,
            confidence=1,
            evidence="Operator pasted it.",
        ),
        MemoryProposal(
            kind=MemoryKind.SEMANTIC,
            scope="project",
            content="The maintainer email is person@example.com.",
            importance=0.8,
            confidence=1,
            evidence="README",
        ),
        MemoryProposal(
            kind=MemoryKind.PROCEDURAL,
            scope="project",
            content="Use the deployment credential configured by the operator.",
            importance=0.8,
            confidence=1,
            evidence="PASSWORD=not-safe-to-persist-123456",
        ),
    ]

    results = [
        consolidator.consolidate(proposal, authorization=MemoryAuthorization.OPERATOR)
        for proposal in proposals
    ]

    assert all(result.outcome is ConsolidationOutcome.REJECTED for result in results)
    assert store.stats().active == 0
    operations = store.operations(limit=10)
    assert len(operations) == 3
    assert all(operation.outcome == "rejected" for operation in operations)
    assert all("secret-value" not in operation.reason for operation in operations)


def test_low_confidence_missing_evidence_and_transient_content_are_rejected(tmp_path: Path) -> None:
    store, consolidator = build_consolidator(tmp_path)
    proposals = [
        MemoryProposal(
            kind=MemoryKind.SEMANTIC,
            scope="project",
            content="The parser probably supports COBOL.",
            importance=0.5,
            confidence=0.2,
            evidence="Guess",
        ),
        MemoryProposal(
            kind=MemoryKind.PROCEDURAL,
            scope="project",
            content="Always run the formatter before tests.",
            importance=0.7,
            confidence=0.9,
        ),
        MemoryProposal(
            kind=MemoryKind.EPISODIC,
            scope="project",
            content="For this one task, the operator is currently on a phone.",
            importance=0.7,
            confidence=0.9,
            evidence="Current chat",
        ),
    ]

    outcomes = {
        consolidator.consolidate(item, authorization=MemoryAuthorization.OPERATOR).outcome
        for item in proposals
    }

    assert outcomes == {ConsolidationOutcome.REJECTED}
    assert store.stats().active == 0


def test_episodic_memory_gets_default_ttl_and_duplicate_is_idempotent(tmp_path: Path) -> None:
    store, consolidator = build_consolidator(tmp_path)
    before = datetime.now(UTC)
    proposal = MemoryProposal(
        kind=MemoryKind.EPISODIC,
        scope="role:reviewer",
        content="Regression test caught the Windows path parsing bug.",
        importance=0.7,
        confidence=0.9,
        evidence="tests/test_paths.py failed before the fix and passed afterward.",
        source_run_id="run-2",
        source_agent="reviewer",
    )

    first = consolidator.consolidate(proposal, authorization=MemoryAuthorization.GOVERNED_TOOL)
    second = consolidator.consolidate(proposal, authorization=MemoryAuthorization.GOVERNED_TOOL)

    assert first.outcome is ConsolidationOutcome.CREATED
    assert second.outcome is ConsolidationOutcome.DUPLICATE
    assert first.item is not None and first.item.expires_at is not None
    assert before + timedelta(days=89) < first.item.expires_at < before + timedelta(days=91)
    assert store.stats().active == 1


def test_similar_memory_is_reported_but_not_silently_overwritten(tmp_path: Path) -> None:
    store, consolidator = build_consolidator(tmp_path)
    existing = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The parser accepts Windows drive-letter paths.",
        importance=0.8,
        confidence=1,
    ).item
    proposal = MemoryProposal(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The parser accepts Windows UNC paths.",
        importance=0.8,
        confidence=0.9,
        evidence="Verified by tests/test_windows_paths.py.",
    )

    result = consolidator.consolidate(proposal, authorization=MemoryAuthorization.OPERATOR)

    assert result.outcome is ConsolidationOutcome.CREATED
    assert existing.memory_id in result.similar_memory_ids
    assert store.stats().active == 2

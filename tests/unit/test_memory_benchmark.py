"""Offline structured-memory retrieval benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from caro.memory.benchmark import MemoryBenchmarkCase, load_benchmark_cases, run_memory_benchmark
from caro.memory.models import MemoryKind
from caro.memory.store import MemoryStore


def test_benchmark_reports_hit_rate_recall_and_mrr_without_access_side_effects(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    parser = store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The parser entry point is src/parser.py.",
        importance=0.9,
        confidence=1,
    ).item
    command = store.put(
        kind=MemoryKind.PROCEDURAL,
        scope="project",
        content="Run uv run pytest -q before committing parser changes.",
        importance=0.8,
        confidence=1,
    ).item
    cases = [
        MemoryBenchmarkCase(
            name="parser-location",
            query="parser entry point",
            scopes=["project"],
            expected_substrings=["src/parser.py"],
        ),
        MemoryBenchmarkCase(
            name="test-command",
            query="pytest parser",
            scopes=["project"],
            expected_substrings=["uv run pytest -q"],
            kinds=[MemoryKind.PROCEDURAL],
        ),
    ]

    report = run_memory_benchmark(store, cases, limit=3, minimum_importance=0)

    assert report.case_count == 2
    assert report.hit_rate == pytest.approx(1)
    assert report.recall_at_k == pytest.approx(1)
    assert report.mean_reciprocal_rank == pytest.approx(1)
    assert store.get(parser.memory_id).access_count == 0
    assert store.get(command.memory_id).access_count == 0


def test_benchmark_scope_prevents_private_role_leakage(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="role:implementer",
        content="Private parser workaround used by implementer.",
        importance=1,
        confidence=1,
    )
    case = MemoryBenchmarkCase(
        name="reviewer-isolation",
        query="parser workaround",
        scopes=["project", "role:reviewer"],
        expected_substrings=[],
        forbidden_substrings=["Private parser workaround"],
    )

    report = run_memory_benchmark(store, [case], limit=5, minimum_importance=0)

    assert report.cases[0].passed is True
    assert report.cases[0].leaked_forbidden == ()


def test_benchmark_yaml_is_strict(tmp_path: Path) -> None:
    dataset = tmp_path / "memory-benchmark.yaml"
    dataset.write_text(
        "cases:\n"
        "  - name: parser\n"
        "    query: parser entry point\n"
        "    scopes: [project]\n"
        "    expected_substrings: [src/parser.py]\n",
        encoding="utf-8",
    )

    cases = load_benchmark_cases(dataset)

    assert cases[0].name == "parser"
    dataset.write_text(
        "cases:\n  - name: bad\n    query: x\n    scopes: [project]\n    unknown: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        load_benchmark_cases(dataset)

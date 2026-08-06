"""Offline, side-effect-free benchmark for structured-memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from caro.memory.models import MemoryKind
from caro.memory.store import MemoryStore


class MemoryBenchmarkCase(BaseModel):
    """One deterministic retrieval and isolation expectation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=2000)
    scopes: list[str] = Field(min_length=1, max_length=16)
    kinds: list[MemoryKind] | None = None
    expected_substrings: list[str] = Field(default_factory=list, max_length=50)
    forbidden_substrings: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_expectations(self) -> MemoryBenchmarkCase:
        """Require a retrieval or isolation assertion."""

        if not self.expected_substrings and not self.forbidden_substrings:
            raise ValueError("benchmark case requires expected_substrings or forbidden_substrings")
        return self


@dataclass(frozen=True, slots=True)
class MemoryBenchmarkCaseResult:
    """Detailed result for one benchmark case."""

    name: str
    passed: bool
    retrieved_memory_ids: tuple[str, ...]
    missing_expected: tuple[str, ...]
    leaked_forbidden: tuple[str, ...]
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class MemoryBenchmarkReport:
    """Aggregate retrieval quality and scope-isolation metrics."""

    case_count: int
    hit_rate: float
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: tuple[MemoryBenchmarkCaseResult, ...]


def _contains(content: str, substring: str) -> bool:
    return substring.casefold() in content.casefold()


def run_memory_benchmark(
    store: MemoryStore,
    cases: list[MemoryBenchmarkCase],
    *,
    limit: int,
    minimum_importance: float,
) -> MemoryBenchmarkReport:
    """Evaluate FTS retrieval without mutating access counters or events."""

    if not cases:
        raise ValueError("memory benchmark requires at least one case")
    results: list[MemoryBenchmarkCaseResult] = []
    expected_total = 0
    expected_found = 0
    reciprocal_total = 0.0
    for case in cases:
        matches = store.search(
            case.query,
            scopes=case.scopes,
            kinds=case.kinds,
            limit=limit,
            min_importance=minimum_importance,
            record_access=False,
        )
        contents = [match.item.content for match in matches]
        missing = tuple(
            expected
            for expected in case.expected_substrings
            if not any(_contains(content, expected) for content in contents)
        )
        leaked = tuple(
            forbidden
            for forbidden in case.forbidden_substrings
            if any(_contains(content, forbidden) for content in contents)
        )
        expected_total += len(case.expected_substrings)
        expected_found += len(case.expected_substrings) - len(missing)
        reciprocal_rank = 1.0 if not case.expected_substrings and not leaked else 0.0
        if case.expected_substrings:
            for index, content in enumerate(contents):
                if any(_contains(content, expected) for expected in case.expected_substrings):
                    reciprocal_rank = 1.0 / (index + 1)
                    break
        reciprocal_total += reciprocal_rank
        results.append(
            MemoryBenchmarkCaseResult(
                name=case.name,
                passed=not missing and not leaked,
                retrieved_memory_ids=tuple(match.item.memory_id for match in matches),
                missing_expected=missing,
                leaked_forbidden=leaked,
                reciprocal_rank=reciprocal_rank,
            )
        )
    passed = sum(result.passed for result in results)
    recall = expected_found / expected_total if expected_total else 1.0
    return MemoryBenchmarkReport(
        case_count=len(results),
        hit_rate=passed / len(results),
        recall_at_k=recall,
        mean_reciprocal_rank=reciprocal_total / len(results),
        cases=tuple(results),
    )


def load_benchmark_cases(path: Path) -> list[MemoryBenchmarkCase]:
    """Load a strict YAML benchmark suite."""

    loaded = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("cases"), list):
        raise ValueError("memory benchmark YAML must contain a cases list")
    try:
        return [MemoryBenchmarkCase.model_validate(item) for item in loaded["cases"]]
    except ValidationError as error:
        raise ValueError(f"invalid memory benchmark case: {error}") from error

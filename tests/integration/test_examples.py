"""Ensure all shipped YAML examples stay valid as schemas evolve."""

from __future__ import annotations

from pathlib import Path

from caro.config import load_settings
from caro.memory.benchmark import load_benchmark_cases


def test_all_example_configs_validate() -> None:
    paths = [Path("caro.example.yaml"), *sorted(Path("examples").glob("*.yaml"))]
    assert paths
    for path in paths:
        settings = load_settings(path)
        assert settings.routing.default in settings.routing.models


def test_readme_documents_terminal_provider_and_langgraph_surfaces() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for phrase in (
        "caro run",
        "deepseek:deepseek-v4-flash",
        "openai:gpt-5.6-sol",
        "caro graph history",
        "caro graph fork",
        "caro graph replay",
        "caro memory status",
        "caro agents",
        "examples/caro/multi-agent.yaml",
        "max_parallel_agents",
        "memory=",
        "ModelFallbackMiddleware",
        "caro memory remember",
        "caro memory benchmark",
        "memory_propose",
        "memory.sqlite3",
        "程序性记忆",
    ):
        assert phrase in readme


def test_memory_benchmark_example_is_valid() -> None:
    cases = load_benchmark_cases(Path("examples/caro/benchmarks/project-memory.yaml"))
    assert len(cases) >= 2

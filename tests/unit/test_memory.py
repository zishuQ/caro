"""Project-scoped long-term memory tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from caro.config import MemorySettings
from caro.errors import ProjectMemoryError
from caro.memory import DEFAULT_MEMORY_TEMPLATE, ProjectMemory


def test_prepare_creates_default_memory_and_returns_backend_path(tmp_path: Path) -> None:
    manager = ProjectMemory(tmp_path, MemorySettings())

    assert manager.prepare() == ("/.caro/memory/AGENTS.md",)
    memory_path = tmp_path / ".caro" / "memory" / "AGENTS.md"
    assert memory_path.read_text(encoding="utf-8") == DEFAULT_MEMORY_TEMPLATE
    assert manager.statuses()[0].size_bytes == len(DEFAULT_MEMORY_TEMPLATE.encode("utf-8"))


def test_prepare_preserves_existing_memory(tmp_path: Path) -> None:
    memory_path = tmp_path / "project" / "AGENTS.md"
    memory_path.parent.mkdir()
    memory_path.write_text("# Existing\n\nKeep this.\n", encoding="utf-8")
    manager = ProjectMemory(tmp_path, MemorySettings(sources=["project/AGENTS.md"]))

    assert manager.prepare() == ("/project/AGENTS.md",)
    assert memory_path.read_text(encoding="utf-8") == "# Existing\n\nKeep this.\n"


def test_disabled_memory_does_not_create_files(tmp_path: Path) -> None:
    manager = ProjectMemory(tmp_path, MemorySettings(enabled=False))

    assert manager.prepare() == ()
    assert not (tmp_path / ".caro" / "memory" / "AGENTS.md").exists()


def test_missing_memory_can_fail_closed(tmp_path: Path) -> None:
    manager = ProjectMemory(tmp_path, MemorySettings(initialize_missing=False))

    with pytest.raises(ProjectMemoryError, match="initialize_missing is false"):
        manager.prepare()


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "memory").symlink_to(outside, target_is_directory=True)
    manager = ProjectMemory(workspace, MemorySettings(sources=["memory/AGENTS.md"]))

    with pytest.raises(ProjectMemoryError, match="outside the workspace"):
        manager.prepare()


def test_oversized_or_non_utf8_memory_is_rejected(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.md"
    oversized.write_text("x" * 1025, encoding="utf-8")
    binary = tmp_path / "binary.md"
    binary.write_bytes(b"\xff\xfe")

    with pytest.raises(ProjectMemoryError, match="exceeds max_file_bytes"):
        ProjectMemory(
            tmp_path,
            MemorySettings(sources=["oversized.md"], max_file_bytes=1024),
        ).prepare()
    with pytest.raises(ProjectMemoryError, match="UTF-8"):
        ProjectMemory(tmp_path, MemorySettings(sources=["binary.md"])).prepare()


def test_operator_memory_append_is_bounded_and_selectable(tmp_path: Path) -> None:
    settings = MemorySettings(sources=["one/AGENTS.md", "two/AGENTS.md"])
    manager = ProjectMemory(tmp_path, settings)

    with pytest.raises(ProjectMemoryError, match="select a memory source"):
        manager.append_operator_memory("Use pytest.")

    status = manager.append_operator_memory("Use   pytest -q\nfor fast checks.", "two/AGENTS.md")

    assert status.source.configured_path == "two/AGENTS.md"
    assert "Use pytest -q for fast checks." in (tmp_path / "two" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Use pytest" not in (tmp_path / "one" / "AGENTS.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("source", ["../AGENTS.md", "/AGENTS.md", "./AGENTS.md", "memory\\AGENTS.md"])
def test_invalid_memory_source_is_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        MemorySettings(sources=[source])

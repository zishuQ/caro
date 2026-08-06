"""Tool approval preview tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from caro.tools.preview import (
    FilePreviewGuardMiddleware,
    build_tool_preview,
    capture_command_snapshot,
    capture_file_snapshot,
    verify_command_snapshot,
    verify_file_snapshot,
)
from caro.tools.registry import ToolRegistry


def test_write_preview_contains_unified_diff(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("before\n", encoding="utf-8")
    registry = ToolRegistry.default()
    assessment = registry.assess(
        "write_file",
        {"file_path": str(target), "content": "after\n"},
    )

    preview = build_tool_preview(
        "write_file",
        {"file_path": str(target), "content": "after\n"},
        tmp_path,
        assessment,
    )

    assert "Risk: HIGH" in preview
    assert "-before" in preview
    assert "+after" in preview


def test_virtual_absolute_file_paths_are_scoped_to_workspace(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("before\n", encoding="utf-8")
    registry = ToolRegistry.default()
    args = {"file_path": "/README.md", "content": "after\n"}
    preview = build_tool_preview("write_file", args, tmp_path, registry.assess("write_file", args))
    assert "+after" in preview


def test_edit_preview_rejects_ambiguous_replacement(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("value\nvalue\n", encoding="utf-8")
    registry = ToolRegistry.default()
    args = {
        "file_path": str(target),
        "old_string": "value",
        "new_string": "updated",
        "replace_all": False,
    }

    preview = build_tool_preview("edit_file", args, tmp_path, registry.assess("edit_file", args))

    assert "occurs 2 times" in preview
    assert "would fail" in preview


def test_file_snapshot_detects_changes_after_preview(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("before\n", encoding="utf-8")
    snapshot = capture_file_snapshot(
        "write_file", {"file_path": str(target), "content": "after\n"}, tmp_path
    )
    assert snapshot is not None
    target.write_text("changed by another process\n", encoding="utf-8")
    reason = verify_file_snapshot(
        snapshot.as_mapping(),
        "write_file",
        {"file_path": str(target), "content": "after\n"},
        tmp_path,
    )
    assert reason is not None


def test_file_snapshot_rejects_changed_operation_arguments(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("before\n", encoding="utf-8")
    snapshot = capture_file_snapshot(
        "write_file", {"file_path": str(target), "content": "approved\n"}, tmp_path
    )
    assert snapshot is not None
    reason = verify_file_snapshot(
        snapshot.as_mapping(),
        "write_file",
        {"file_path": str(target), "content": "changed-by-edited-approval\n"},
        tmp_path,
    )
    assert reason == "the approved file arguments changed after the preview"


def test_file_preview_guard_blocks_changed_checkpoint_tool_call(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("before\n", encoding="utf-8")
    guard = FilePreviewGuardMiddleware(tmp_path)
    call = {
        "name": "write_file",
        "args": {"file_path": str(target), "content": "after\n"},
        "id": "call-1",
        "type": "tool_call",
    }

    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import AIMessage

    response = guard.wrap_model_call(
        ModelRequest(model=FakeListChatModel(responses=["ok"]), messages=[]),
        lambda _: ModelResponse(result=[AIMessage(content="", tool_calls=[call])]),
    )
    assert isinstance(response, ModelResponse)
    target.write_text("changed\n", encoding="utf-8")
    request = type(
        "Request",
        (),
        {"tool_call": response.result[0].tool_calls[0]},
    )()
    with pytest.raises(RuntimeError, match="changed after the preview"):
        guard._verify(request)


def test_git_commit_snapshot_detects_staged_changes(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    target = tmp_path / "README.md"
    target.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    snapshot = capture_command_snapshot("execute", {"command": "git commit -m test"}, tmp_path)
    assert snapshot is not None
    target.write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    guard = FilePreviewGuardMiddleware(tmp_path)
    call = {
        "name": "execute",
        "args": {"command": "git commit -m test"},
        "id": "git-call",
        "type": "tool_call",
        "_caro_command_snapshot": snapshot,
    }
    request = type("Request", (), {"tool_call": call})()
    with pytest.raises(RuntimeError, match="staged Git snapshot"):
        guard._verify(request)


def test_git_commit_snapshot_binds_command_and_rejects_repo_switches(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    target = tmp_path / "README.md"
    target.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    snapshot = capture_command_snapshot("execute", {"command": "git commit -m test"}, tmp_path)
    assert snapshot is not None
    assert verify_command_snapshot(
        snapshot,
        "execute",
        {"command": "git commit -m changed"},
        tmp_path,
    ) == "the approved command was changed after the preview"
    assert capture_command_snapshot(
        "execute", {"command": "git -C subrepo commit -m test"}, tmp_path
    ) is None

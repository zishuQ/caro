"""Human-readable previews for Caro's high-risk tool calls."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage

from caro.tools.policy import ToolAssessment

_MAX_PREVIEW_CHARS = 24_000
_SNAPSHOT_KEY = "_caro_preview_snapshot"
_COMMAND_SNAPSHOT_KEY = "_caro_command_snapshot"


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Metadata captured before a side-effect approval is displayed."""

    path: str
    exists: bool
    kind: str | None
    fingerprint: str | None
    arguments_digest: str | None = None

    def as_mapping(self) -> dict[str, object]:
        """Return a checkpoint-safe representation."""

        return {
            "path": self.path,
            "exists": self.exists,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "arguments_digest": self.arguments_digest,
        }


def _workspace_path(value: object, workspace: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    elif not candidate.is_relative_to(workspace.resolve()):
        # DeepAgents virtual backends expose workspace paths with a leading
        # slash (for example `/src/main.py`). Map those virtual paths into the
        # physical workspace instead of treating them as host absolute paths.
        candidate = workspace / candidate.relative_to(candidate.anchor)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        return None
    return resolved


def _path_fingerprint(path: Path) -> tuple[str, str] | None:
    """Return a bounded fingerprint for an existing file or directory."""

    try:
        stat = path.stat()
    except OSError:
        return None
    digest = hashlib.sha256()
    if path.is_file():
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        return "file", digest.hexdigest()
    if path.is_dir():
        try:
            entries: list[str] = []
            for root, directories, files in os.walk(path, followlinks=False):
                directories.sort()
                files.sort()
                relative_root = Path(root).relative_to(path)
                entries.extend(
                    f"{relative_root / name}:d" for name in directories
                )
                entries.extend(f"{relative_root / name}:f" for name in files)
            digest.update("\n".join(entries[:10_000]).encode())
        except OSError:
            return None
        digest.update(str(stat.st_mtime_ns).encode())
        return "directory", digest.hexdigest()
    return "other", f"{stat.st_mode}:{stat.st_size}:{stat.st_mtime_ns}"


def capture_file_snapshot(
    name: str, args: Mapping[str, object], workspace: Path
) -> FileSnapshot | None:
    """Capture the state that a file preview was based on."""

    if name not in {"write_file", "edit_file", "delete"}:
        return None
    path = _workspace_path(args.get("file_path", args.get("path")), workspace)
    if path is None:
        return None
    fingerprint = _path_fingerprint(path)
    return FileSnapshot(
        path=str(path),
        exists=fingerprint is not None,
        kind=None if fingerprint is None else fingerprint[0],
        fingerprint=None if fingerprint is None else fingerprint[1],
        arguments_digest=_arguments_digest(name, args),
    )


def _arguments_digest(name: str, args: Mapping[str, object]) -> str | None:
    """Hash the exact JSON-shaped operation arguments for approval binding."""

    try:
        encoded = json.dumps(
            {"name": name, "args": args},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def verify_file_snapshot(
    snapshot: Mapping[str, object], name: str, args: Mapping[str, object], workspace: Path
) -> str | None:
    """Return a reason when the approved file target changed after preview."""

    path = _workspace_path(args.get("file_path", args.get("path")), workspace)
    expected_path = snapshot.get("path")
    if path is None or not isinstance(expected_path, str) or str(path) != expected_path:
        return "the approved file target no longer resolves to the same workspace path"
    expected_arguments_digest = snapshot.get("arguments_digest")
    current_arguments_digest = _arguments_digest(name, args)
    if (
        not isinstance(expected_arguments_digest, str)
        or current_arguments_digest is None
        or current_arguments_digest != expected_arguments_digest
    ):
        return "the approved file arguments changed after the preview"
    current = _path_fingerprint(path)
    expected_exists = snapshot.get("exists") is True
    if (current is not None) != expected_exists:
        return "the approved file was created or removed after the preview"
    if current is None:
        return None
    if current[0] != snapshot.get("kind") or current[1] != snapshot.get("fingerprint"):
        return "the approved file changed after the preview; review the new diff"
    return None


def _git_commit_tokens(command: object) -> list[str] | None:
    if not isinstance(command, str):
        return None
    try:
        return shlex.split(command)
    except ValueError:
        return None


def _git_commit_command(command: object) -> bool:
    tokens = _git_commit_tokens(command)
    return bool(
        tokens
        and Path(tokens[0]).name.casefold() == "git"
        and any(token.casefold() == "commit" for token in tokens[1:])
    )


def _git_commit_scope_supported(command: object) -> bool:
    """Return whether a commit command targets the configured workspace repo."""

    tokens = _git_commit_tokens(command)
    if not tokens or Path(tokens[0]).name.casefold() != "git":
        return False
    try:
        commit_index = next(index for index, token in enumerate(tokens) if token.casefold() == "commit")
    except StopIteration:
        return False

    path_switches = ("--git-dir", "--work-tree", "--super-prefix")
    for token in tokens[1:commit_index]:
        lowered = token.casefold()
        if token == "-C" or token.startswith("-C"):
            return False
        if lowered in path_switches or any(lowered.startswith(f"{switch}=") for switch in path_switches):
            return False

    value_options = {
        "-m",
        "--message",
        "-f",
        "--file",
        "--author",
        "--date",
        "--cleanup",
        "--trailer",
        "-s",
        "--gpg-sign",
    }
    after_commit = tokens[commit_index + 1 :]
    expect_value = False
    for token in after_commit:
        if expect_value:
            expect_value = False
            continue
        if token == "--" or not token.startswith("-"):
            return False
        lowered = token.casefold()
        if lowered in {"--pathspec-from-file", "--pathspec-file-nul"} or lowered.startswith(
            "--pathspec-from-file="
        ):
            return False
        if lowered in value_options:
            expect_value = True
    return not expect_value


def _staged_diff_digest(workspace: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=workspace,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def capture_command_snapshot(
    name: str, args: Mapping[str, object], workspace: Path
) -> dict[str, object] | None:
    """Capture the staged snapshot used by an approved Git commit."""

    command = args.get("command")
    if (
        name != "execute"
        or not _git_commit_command(command)
        or not _git_commit_scope_supported(command)
        or not isinstance(command, str)
    ):
        return None
    digest = _staged_diff_digest(workspace)
    if digest is None:
        return None
    return {
        "kind": "git-commit",
        "digest": digest,
        "command_digest": hashlib.sha256(command.encode("utf-8")).hexdigest(),
    }


def verify_command_snapshot(
    snapshot: Mapping[str, object], name: str, args: Mapping[str, object], workspace: Path
) -> str | None:
    """Return a reason when staged Git content changed after preview."""

    if name != "execute" or snapshot.get("kind") != "git-commit":
        return "the approved command no longer matches the preview"
    command = args.get("command")
    if not _git_commit_command(command) or not _git_commit_scope_supported(command):
        return "the approved command was changed after the preview"
    if not isinstance(command, str):
        return "the approved command was changed after the preview"
    expected_command_digest = snapshot.get("command_digest")
    actual_command_digest = hashlib.sha256(command.encode("utf-8")).hexdigest()
    if not isinstance(expected_command_digest, str) or actual_command_digest != expected_command_digest:
        return "the approved command was changed after the preview"
    digest = _staged_diff_digest(workspace)
    if digest is None or digest != snapshot.get("digest"):
        return "the staged Git snapshot changed after the preview"
    return None


class FilePreviewGuardMiddleware(AgentMiddleware[Any, Any, Any]):
    """Persist preview snapshots in tool calls and reject TOCTOU changes."""

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace.expanduser().resolve()

    @staticmethod
    def _attach(
        response: ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any],
        workspace: Path,
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        if isinstance(response, ExtendedModelResponse):
            attached = FilePreviewGuardMiddleware._attach(response.model_response, workspace)
            if not isinstance(attached, ModelResponse):
                return response
            return ExtendedModelResponse(model_response=attached, command=response.command)
        messages = [response] if isinstance(response, AIMessage) else response.result
        changed = False
        copied: list[Any] = []
        for message in messages:
            if not isinstance(message, AIMessage) or not message.tool_calls:
                copied.append(message)
                continue
            message_copy = message.model_copy(deep=True)
            for call in message_copy.tool_calls:
                name = call.get("name")
                args = call.get("args")
                if isinstance(name, str) and isinstance(args, Mapping):
                    snapshot = capture_file_snapshot(name, args, workspace)
                    if snapshot is not None:
                        cast("dict[str, Any]", call)[_SNAPSHOT_KEY] = snapshot.as_mapping()
                        changed = True
                    command_snapshot = capture_command_snapshot(name, args, workspace)
                    if command_snapshot is not None:
                        cast("dict[str, Any]", call)[_COMMAND_SNAPSHOT_KEY] = command_snapshot
                        changed = True
            copied.append(message_copy)
        if isinstance(response, AIMessage):
            return cast("AIMessage", copied[0])
        if not changed:
            return response
        return ModelResponse(result=copied, structured_response=response.structured_response)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Capture snapshots before the model result enters graph state."""

        return self._attach(handler(request), self.workspace)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Async counterpart of :meth:`wrap_model_call`."""

        return self._attach(await handler(request), self.workspace)

    def _verify(self, request: Any) -> None:
        snapshot = request.tool_call.get(_SNAPSHOT_KEY)
        command_snapshot = request.tool_call.get(_COMMAND_SNAPSHOT_KEY)
        if isinstance(command_snapshot, Mapping):
            name = request.tool_call.get("name")
            args = request.tool_call.get("args")
            if not isinstance(name, str) or not isinstance(args, Mapping):
                raise RuntimeError("Caro could not validate the approved command")
            reason = verify_command_snapshot(command_snapshot, name, args, self.workspace)
            if reason is not None:
                raise RuntimeError(f"Caro blocked the command: {reason}")
            return
        if request.tool_call.get("name") == "execute" and _git_commit_command(
            request.tool_call.get("args", {}).get("command")
            if isinstance(request.tool_call.get("args"), Mapping)
            else None
        ):
            raise RuntimeError(
                "Caro blocked the Git commit because no approval snapshot was preserved"
            )
        if not isinstance(snapshot, Mapping):
            if request.tool_call.get("name") in {"write_file", "edit_file", "delete"}:
                raise RuntimeError(
                    "Caro blocked the file operation because no approval snapshot was preserved"
                )
            return
        name = request.tool_call.get("name")
        args = request.tool_call.get("args")
        if not isinstance(name, str) or not isinstance(args, Mapping):
            raise RuntimeError("Caro could not validate the approved file operation")
        reason = verify_file_snapshot(snapshot, name, args, self.workspace)
        if reason is not None:
            raise RuntimeError(f"Caro blocked the file operation: {reason}")

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Verify the checkpointed snapshot immediately before execution."""

        self._verify(request)
        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        """Async counterpart of :meth:`wrap_tool_call`."""

        self._verify(request)
        return await handler(request)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="replace")


def _bounded(value: str) -> str:
    if len(value) <= _MAX_PREVIEW_CHARS:
        return value
    marker = "\n... [preview truncated by Caro] ...\n"
    return value[: max(0, _MAX_PREVIEW_CHARS - len(marker))] + marker


def _diff(path: Path, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    result = "".join(lines)
    return _bounded(result or "(no textual change)")


def _file_preview(name: str, args: Mapping[str, object], workspace: Path) -> str:
    raw_path = args.get("file_path", args.get("path"))
    path = _workspace_path(raw_path, workspace)
    if path is None:
        return "Path is missing or outside Caro's configured workspace."

    if name == "delete":
        if not path.exists():
            return f"Delete target does not exist: {path}"
        if path.is_dir():
            try:
                children = sorted(str(item.relative_to(path)) for item in path.rglob("*"))
            except OSError as error:
                return f"Delete directory {path}\nUnable to enumerate contents: {error}"
            listing = "\n".join(children[:200]) or "(empty directory)"
            if len(children) > 200:
                listing += f"\n... and {len(children) - 200} more entries"
            return f"Delete directory: {path}\nContents:\n{listing}"
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        return f"Delete file: {path}\nSize: {size if size >= 0 else 'unknown'} bytes"

    before = _read_text(path) if path.is_file() else ""
    if name == "write_file":
        content = args.get("content")
        if not isinstance(content, str):
            return f"Write target: {path}\nContent is missing or not text."
        return f"Write target: {path}\n\n{_diff(path, before, content)}"

    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return f"Edit target: {path}\nold_string and new_string must both be text."
    if old == new:
        return f"Edit target: {path}\nNo change: old_string equals new_string."
    occurrences = before.count(old)
    replace_all = bool(args.get("replace_all", False))
    if occurrences == 0:
        return f"Edit target: {path}\nold_string was not found; the tool call will fail."
    if occurrences > 1 and not replace_all:
        return (
            f"Edit target: {path}\nold_string occurs {occurrences} times; "
            "replace_all=false would fail."
        )
    after = before.replace(old, new, -1 if replace_all else 1)
    return f"Edit target: {path}\n\n{_diff(path, before, after)}"


def _staged_git_preview(command: object, workspace: Path) -> str | None:
    """Return a read-only staged diff for a direct ``git commit`` command."""

    if not isinstance(command, str):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or tokens[0] != "git" or "commit" not in tokens[1:]:
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Unable to collect staged diff: {error}"
    if result.returncode != 0:
        return f"Unable to collect staged diff: {result.stderr.strip() or 'git diff failed'}"
    return _bounded(result.stdout or "(no staged changes)")


def build_tool_preview(
    name: str,
    args: Mapping[str, object] | None,
    workspace: Path,
    assessment: ToolAssessment,
) -> str:
    """Build a bounded preview appended to a Caro HITL prompt.

    Args:
        name: Canonical model-visible tool name.
        args: Tool arguments supplied by the model.
        workspace: Configured workspace boundary.
        assessment: Invocation-specific risk assessment.

    Returns:
        Human-readable risk metadata and a concrete side-effect preview.
    """

    values = args or {}
    lines = [f"Risk: {assessment.risk.value.upper()}", f"Reason: {assessment.reason}"]
    lines.extend(f"{label}: {value}" for label, value in assessment.metadata)
    if name in {"write_file", "edit_file", "delete"}:
        lines.extend(("", _file_preview(name, values, workspace)))
    elif name == "execute":
        command = values.get("command", "")
        lines.extend(("", "Command:", str(command)))
        if staged := _staged_git_preview(command, workspace):
            lines.extend(("", "Staged diff:", staged))
    else:
        lines.extend(("", "Arguments:", str(dict(values))))
    return _bounded("\n".join(lines))

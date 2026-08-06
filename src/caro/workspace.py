"""Workspace boundary creation and guarded local shell backend."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from caro.errors import UnsafeCommandError
from caro.observability.events import strip_langfuse_environment

_CONTROL_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo|su\s+-|shutdown|reboot|poweroff|halt|mkfs(?:\.[a-z0-9]+)?|fdisk|parted)\b",
    flags=re.IGNORECASE,
)
_GIT_MUTATION_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)git\s+(?:push\b|reset\s+--hard\b|clean\s+-[^\n;&|]*f)",
    flags=re.IGNORECASE,
)
_RM_PATTERN = re.compile(r"(?:^|[;&|]\s*)rm\s+([^;&|]+)", flags=re.IGNORECASE)
_SHELL_WRAPPER_COMMANDS = frozenset(
    {
        "bash",
        "busybox",
        "command",
        "dash",
        "env",
        "exec",
        "fish",
        "find",
        "ionice",
        "nice",
        "nohup",
        "parallel",
        "perl",
        "php",
        "ruby",
        "setsid",
        "sh",
        "stdbuf",
        "timeout",
        "watch",
        "xargs",
        "zsh",
    }
)
_SHELL_PUNCTUATION = frozenset({";", "&", "|", "<", ">"})


@dataclass(frozen=True, slots=True)
class CommandInspection:
    """Static shell policy result."""

    allowed: bool
    reason: str | None = None


class ShellGuard:
    """Conservative command deny-list and workspace-escape detector.

    This is a guardrail, not a kernel/process sandbox. It intentionally rejects
    shell metaprogramming that would make static inspection bypassable.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()

    def inspect(self, command: str) -> CommandInspection:
        """Inspect one shell command without executing it."""

        if not isinstance(command, str) or not command.strip():
            return CommandInspection(False, "command must be a non-empty string")
        if "\x00" in command or "\n" in command or "\r" in command:
            return CommandInspection(False, "multi-line or NUL-containing shell commands are blocked")
        if "$(" in command or "`" in command:
            return CommandInspection(False, "shell command substitution is blocked")
        if "$" in command:
            return CommandInspection(False, "shell environment expansion is blocked")
        if re.search(r"(?:^|[\s=])(?:~|\.\.(?:/|$)|\$\{?HOME\}?)", command):
            return CommandInspection(False, "home-directory and parent-directory escapes are blocked")
        if _CONTROL_COMMAND_PATTERN.search(command):
            return CommandInspection(False, "host administration or disk-management commands are blocked")
        if _GIT_MUTATION_PATTERN.search(command):
            return CommandInspection(
                False,
                "automatic git push or destructive git history/worktree commands are blocked",
            )
        if re.search(r"\bdd\s+[^;&|]*\bof\s*=\s*/dev/", command, flags=re.IGNORECASE):
            return CommandInspection(False, "raw device writes are blocked")
        if re.search(r":\s*\(\s*\)\s*\{", command):
            return CommandInspection(False, "fork-bomb syntax is blocked")

        rm_result = self._inspect_recursive_remove(command)
        if not rm_result.allowed:
            return rm_result
        return self._inspect_tokens(command)

    def assert_safe(self, command: str) -> None:
        """Raise an actionable exception when a command is blocked."""

        inspection = self.inspect(command)
        if not inspection.allowed:
            raise UnsafeCommandError(inspection.reason or "unsafe shell command")

    def _inspect_recursive_remove(self, command: str) -> CommandInspection:
        for match in _RM_PATTERN.finditer(command):
            try:
                arguments = shlex.split(match.group(1))
            except ValueError:
                return CommandInspection(False, "unparseable rm command is blocked")
            recursive = any(argument.startswith("-") and ("r" in argument or "R" in argument) for argument in arguments)
            if not recursive:
                continue
            targets = [argument for argument in arguments if not argument.startswith("-")]
            if not targets:
                return CommandInspection(False, "recursive rm without an explicit target is blocked")
            for target in targets:
                normalized = target.rstrip("/") or "/"
                if normalized in {"/", ".", "*", "./*"} or target.startswith(("/", "~", "..", "$")):
                    return CommandInspection(False, "broad or out-of-workspace recursive deletion is blocked")
        return CommandInspection(True)

    def _inspect_tokens(self, command: str) -> CommandInspection:
        """Reject shell composition and paths that resolve outside the workspace."""

        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return CommandInspection(False, "unparseable shell quoting is blocked")

        if not tokens:
            return CommandInspection(False, "command must contain an executable")
        if any(token in _SHELL_PUNCTUATION for token in tokens):
            return CommandInspection(False, "shell pipelines, redirections, and command chaining are blocked")

        executable = Path(tokens[0]).name.casefold()
        if executable in _SHELL_WRAPPER_COMMANDS:
            return CommandInspection(False, f"shell command wrapper is blocked: {executable}")
        if executable == "git":
            git_operations = {token.casefold() for token in tokens[1:]}
            if any(token.casefold() in {"-c", "--config-env"} for token in tokens[1:]):
                return CommandInspection(False, "git runtime configuration overrides are blocked")
            for index, token in enumerate(tokens[1:], start=1):
                if token == "-C":
                    if index + 1 >= len(tokens):
                        return CommandInspection(False, "git -C requires a workspace path")
                    result = self._inspect_path_token(tokens[index + 1], index=index + 1)
                elif token.startswith("-C"):
                    result = self._inspect_path_token(token[2:], index=index)
                else:
                    continue
                if not result.allowed:
                    return result
            if "push" in git_operations or "reset" in git_operations:
                return CommandInspection(
                    False,
                    "automatic git push and reset operations are blocked",
                )
            if "clean" in git_operations and any(
                token.casefold().startswith("-") and "f" in token.casefold()
                for token in tokens[1:]
            ):
                return CommandInspection(False, "forced git clean operations are blocked")
            if "config" in git_operations and {"--global", "--system"}.intersection(git_operations):
                return CommandInspection(False, "global and system git configuration changes are blocked")
        if executable in {"python", "python3", "python3.12", "node"} and any(
            token in {"-c", "-e", "-", "--command", "--eval"} for token in tokens[1:]
        ):
            return CommandInspection(False, "inline interpreter execution is blocked")

        for index, token in enumerate(tokens):
            if (token in _SHELL_PUNCTUATION or token.startswith("-")) and (
                not token.startswith("-") or "=" not in token
            ):
                continue
            candidates = [token]
            if "=" in token and token.startswith("-"):
                candidates.append(token.partition("=")[2])
            for candidate in candidates:
                result = self._inspect_path_token(candidate, index=index)
                if not result.allowed:
                    return result
        return CommandInspection(True)

    def _inspect_path_token(self, candidate: str, *, index: int) -> CommandInspection:
        """Check one shell token while tolerating ordinary non-path arguments."""

        if not candidate or candidate in {".", "./", "/dev/null"}:
            return CommandInspection(True)
        if "://" in candidate and not candidate.startswith("file:"):
            return CommandInspection(True)
        if candidate.startswith(("~", "$")):
            return CommandInspection(False, "home-directory and environment-based paths are blocked")

        path = Path(candidate)
        if path.is_absolute():
            if index == 0 and os.access(candidate, os.X_OK):
                return CommandInspection(True)
            resolved = path.resolve(strict=False)
        elif "/" in candidate or (self.workspace / path).exists():
            resolved = (self.workspace / path).resolve(strict=False)
        else:
            return CommandInspection(True)
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            return CommandInspection(False, f"path outside workspace is blocked: {candidate}")
        return CommandInspection(True)


def prepare_workspace(root: Path) -> Path:
    """Create and validate the configured workspace directory."""

    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ValueError(f"workspace is not a directory: {resolved}")
    return resolved


def filtered_environment(names: list[str], *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only explicitly allowed environment variables for shell children."""

    source = os.environ if environ is None else environ
    return {name: source[name] for name in names if name in source}


class GuardedShellBackend(LocalShellBackend):
    """DeepAgents local backend with Caro's non-bypassable command guard."""

    def __init__(
        self,
        root_dir: Path,
        *,
        timeout: int = 120,
        allowed_environment: list[str] | None = None,
        inherit_environment: bool = False,
    ) -> None:
        workspace = prepare_workspace(root_dir)
        environment = filtered_environment(allowed_environment or ["PATH", "LANG", "LC_ALL", "TERM"])
        strip_langfuse_environment(environment)
        super().__init__(
            root_dir=workspace,
            virtual_mode=True,
            timeout=timeout,
            env=environment,
            inherit_env=inherit_environment,
        )
        # LocalShellBackend rebuilds ``_env`` from the parent process when
        # inheritance is enabled, so apply the credential boundary again after
        # its initialization.
        strip_langfuse_environment(self._env)
        self.guard = ShellGuard(workspace)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Block unsafe input or execute it inside the configured working directory."""

        inspection = self.guard.inspect(command)
        if not inspection.allowed:
            return ExecuteResponse(
                output=f"Blocked by Caro shell policy: {inspection.reason}",
                exit_code=126,
                truncated=False,
            )
        return super().execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Asynchronously execute through the same guard."""

        return await asyncio.to_thread(self.execute, command, timeout=timeout)


_SANDBOX_SYSTEM_PATHS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
    "/usr/games",
    "/usr/local/games",
)


def _sandbox_path_value(value: str, workspace: Path) -> str:
    """Map only executable paths visible inside the bubblewrap namespace."""

    mapped: list[str] = []
    system_paths = {Path(path) for path in _SANDBOX_SYSTEM_PATHS}
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        entry_path = Path(entry)
        try:
            resolved = entry_path.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        try:
            relative = resolved.relative_to(workspace)
        except ValueError:
            if entry_path in system_paths or resolved in system_paths:
                mapped.append(str(entry_path))
        else:
            mapped.append(str(Path("/workspace") / relative))
    return os.pathsep.join(mapped)


class BubblewrapShellBackend(GuardedShellBackend):
    """Guarded shell execution inside an unprivileged bubblewrap namespace."""

    _BIND_READONLY = ("/usr", "/bin", "/sbin", "/lib", "/lib64")

    def __init__(
        self,
        root_dir: Path,
        *,
        timeout: int = 120,
        allowed_environment: list[str] | None = None,
        inherit_environment: bool = False,
    ) -> None:
        super().__init__(
            root_dir,
            timeout=timeout,
            allowed_environment=allowed_environment,
            inherit_environment=inherit_environment,
        )
        self._bubblewrap = shutil.which("bwrap")
        if self._bubblewrap is None:
            raise RuntimeError(
                "workspace.sandbox_backend='bubblewrap' requires the bwrap executable"
            )

    def _command(self, command: str) -> str:
        """Build a namespace command without interpolating shell input."""

        assert self._bubblewrap is not None
        arguments = [
            self._bubblewrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
        ]
        for path in self._BIND_READONLY:
            if Path(path).exists():
                arguments.extend(("--ro-bind", path, path))
        arguments.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/home",
                "--dir",
                "/tmp/caro-bin",
            )
        )
        for alias in ("python", "python3"):
            if Path("/usr/bin/python3").exists():
                arguments.extend(("--symlink", "/usr/bin/python3", f"/tmp/caro-bin/{alias}"))
        arguments.extend(
            (
                "--bind",
                str(self.guard.workspace),
                "/workspace",
                "--chdir",
                "/workspace",
                "/bin/sh",
                "-c",
                command,
            )
        )
        return shlex.join(arguments)

    def _environment(self) -> dict[str, str]:
        """Return a namespace-safe copy of the curated child environment."""

        environment = dict(self._env)
        workspace = self.guard.workspace
        path_entries = ["/tmp/caro-bin"]
        project_bin = workspace / ".venv" / "bin"
        if project_bin.is_dir():
            path_entries.append("/workspace/.venv/bin")
        if "PATH" in environment:
            mapped_path = _sandbox_path_value(environment["PATH"], workspace)
            if mapped_path:
                path_entries.extend(mapped_path.split(os.pathsep))
        path_entries.extend(_SANDBOX_SYSTEM_PATHS)
        environment["PATH"] = os.pathsep.join(dict.fromkeys(path_entries))
        for name in ("PYTHONPATH", "VIRTUAL_ENV"):
            if name in environment:
                environment[name] = _sandbox_path_value(environment[name], workspace)
        for name in ("BASH_ENV", "ENV", "PYTHONSTARTUP", "RUBYOPT", "PERL5OPT"):
            environment.pop(name, None)
        environment["HOME"] = "/tmp/home"
        environment["PWD"] = "/workspace"
        environment["OLDPWD"] = "/workspace"
        return environment

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Inspect and execute a command inside the bubblewrap namespace."""

        inspection = self.guard.inspect(command)
        if not inspection.allowed:
            return ExecuteResponse(
                output=f"Blocked by Caro shell policy: {inspection.reason}",
                exit_code=126,
                truncated=False,
            )
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")
        try:
            result = subprocess.run(
                self._command(command),
                check=False,
                shell=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=effective_timeout,
                env=self._environment(),
                cwd=str(self.cwd),
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout} seconds",
                exit_code=124,
                truncated=False,
            )
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.extend(f"[stderr] {line}" for line in result.stderr.strip().split("\n"))
        output = "\n".join(output_parts) if output_parts else "<no output>"
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes] + (
                f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            )
            truncated = True
        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
        return ExecuteResponse(output=output, exit_code=result.returncode, truncated=truncated)

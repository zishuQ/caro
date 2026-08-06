"""Project-scoped long-term memory backed by governed workspace files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from caro.config import MemorySettings
from caro.errors import ProjectMemoryError

DEFAULT_MEMORY_TEMPLATE = """# Caro Project Memory

<!--
This file is durable project-scoped memory loaded by DeepAgents.
Keep entries concise, verifiable, and useful across future tasks.
Never store credentials, access tokens, passwords, or transient task details.
-->

## Project facts

- No confirmed project facts have been recorded yet.

## Conventions and decisions

- No confirmed conventions or decisions have been recorded yet.

## Proven commands and pitfalls

- No proven commands or pitfalls have been recorded yet.
"""

PROJECT_MEMORY_SYSTEM_INSTRUCTIONS = (
    "Caro project-memory policy:\n"
    "- Project memory is scoped to this code workspace, not to a person.\n"
    "- Persist only verified project facts, stable conventions, architecture decisions, proven commands, "
    "and reusable pitfalls.\n"
    "- Do not persist personal identifiers, credentials, secrets, transient requests, unverified guesses, "
    "or raw model output.\n"
    "- Prefer concise edits to the configured memory files and let Caro's normal file-write approval policy "
    "govern them."
)


@dataclass(frozen=True, slots=True)
class ProjectMemorySource:
    """Resolved host and backend paths for one configured memory file."""

    configured_path: str
    host_path: Path
    backend_path: str


@dataclass(frozen=True, slots=True)
class ProjectMemoryStatus:
    """Inspectable state for one project-memory source."""

    source: ProjectMemorySource
    exists: bool
    size_bytes: int


class ProjectMemory:
    """Initialize and inspect DeepAgents memory without escaping the workspace."""

    def __init__(self, workspace: Path, settings: MemorySettings) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.settings = settings

    def sources(self) -> tuple[ProjectMemorySource, ...]:
        """Resolve configured sources to safe host and virtual backend paths."""

        resolved: list[ProjectMemorySource] = []
        for configured_path in self.settings.sources:
            candidate = self.workspace.joinpath(*configured_path.split("/"))
            resolved_target = candidate.resolve()
            try:
                resolved_target.relative_to(self.workspace)
            except ValueError as error:
                raise ProjectMemoryError(
                    f"memory source resolves outside the workspace: {configured_path} -> {resolved_target}"
                ) from error
            resolved.append(
                ProjectMemorySource(
                    configured_path=configured_path,
                    host_path=candidate,
                    backend_path=f"/{configured_path}",
                )
            )
        return tuple(resolved)

    def _validate_existing(self, source: ProjectMemorySource) -> int:
        path = source.host_path
        if path.is_symlink():
            raise ProjectMemoryError(f"memory source cannot be a symbolic link: {source.configured_path}")
        if not path.is_file():
            raise ProjectMemoryError(f"memory source is not a regular file: {source.configured_path}")
        size = path.stat().st_size
        if size > self.settings.max_file_bytes:
            raise ProjectMemoryError(
                f"memory source exceeds max_file_bytes ({size} > {self.settings.max_file_bytes}): "
                f"{source.configured_path}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ProjectMemoryError(f"memory source must be UTF-8: {source.configured_path}") from error
        if "\x00" in content:
            raise ProjectMemoryError(f"memory source cannot contain NUL bytes: {source.configured_path}")
        return size

    def statuses(self) -> tuple[ProjectMemoryStatus, ...]:
        """Return source status without creating missing files."""

        statuses: list[ProjectMemoryStatus] = []
        for source in self.sources():
            if source.host_path.exists() or source.host_path.is_symlink():
                size = self._validate_existing(source)
                statuses.append(ProjectMemoryStatus(source=source, exists=True, size_bytes=size))
            else:
                statuses.append(ProjectMemoryStatus(source=source, exists=False, size_bytes=0))
        return tuple(statuses)

    def prepare(self) -> tuple[str, ...]:
        """Create missing memory templates and return DeepAgents backend paths."""

        if not self.settings.enabled:
            return ()
        backend_paths: list[str] = []
        for source in self.sources():
            if not source.host_path.exists() and not source.host_path.is_symlink():
                if not self.settings.initialize_missing:
                    raise ProjectMemoryError(
                        f"memory source is missing and initialize_missing is false: {source.configured_path}"
                    )
                source.host_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with source.host_path.open("x", encoding="utf-8", errors="strict") as stream:
                        stream.write(DEFAULT_MEMORY_TEMPLATE)
                except FileExistsError:
                    pass
                except OSError as error:
                    raise ProjectMemoryError(f"cannot initialize memory source: {source.configured_path}") from error
            self._validate_existing(source)
            backend_paths.append(source.backend_path)
        return tuple(backend_paths)

    def read(self, configured_path: str | None = None) -> tuple[tuple[ProjectMemorySource, str], ...]:
        """Read all configured sources, or one selected source."""

        selected = self._select(configured_path)
        result: list[tuple[ProjectMemorySource, str]] = []
        for source in selected:
            if not source.host_path.exists():
                raise ProjectMemoryError(f"memory source is not initialized: {source.configured_path}")
            self._validate_existing(source)
            result.append((source, source.host_path.read_text(encoding="utf-8")))
        return tuple(result)

    def append_operator_memory(self, text: str, configured_path: str | None = None) -> ProjectMemoryStatus:
        """Append one explicit operator-provided durable learning."""

        normalized = " ".join(text.split())
        if not normalized:
            raise ProjectMemoryError("memory text cannot be empty")
        if len(normalized) > 4000:
            raise ProjectMemoryError("one memory entry cannot exceed 4000 characters")
        self.prepare()
        selected = self._select(configured_path)
        if len(selected) != 1:
            raise ProjectMemoryError("select a memory source when more than one source is configured")
        source = selected[0]
        current = source.host_path.read_text(encoding="utf-8").rstrip()
        timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
        updated = f"{current}\n\n- [{timestamp}] {normalized}\n"
        size = len(updated.encode("utf-8"))
        if size > self.settings.max_file_bytes:
            raise ProjectMemoryError(
                f"memory update would exceed max_file_bytes ({size} > {self.settings.max_file_bytes})"
            )
        source.host_path.write_text(updated, encoding="utf-8")
        return ProjectMemoryStatus(source=source, exists=True, size_bytes=size)

    def _select(self, configured_path: str | None) -> tuple[ProjectMemorySource, ...]:
        sources = self.sources()
        if configured_path is None:
            return sources
        for source in sources:
            if source.configured_path == configured_path:
                return (source,)
        raise ProjectMemoryError(f"memory source is not configured: {configured_path}")

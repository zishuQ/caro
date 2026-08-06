"""Typed access to LangGraph SQLite checkpoints, history, replay, and forks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import StateSnapshot


def graph_config(
    thread_id: str,
    *,
    checkpoint_id: str | None = None,
    recursion_limit: int | None = None,
) -> RunnableConfig:
    """Build a correctly shaped LangGraph thread/checkpoint config."""

    configurable: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    config: RunnableConfig = {"configurable": configurable}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def checkpoint_id(config: RunnableConfig | None) -> str | None:
    """Read a checkpoint ID from a LangGraph config."""

    if config is None:
        return None
    value = config.get("configurable", {}).get("checkpoint_id")
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class CheckpointView:
    """CLI-safe checkpoint metadata without dumping graph state."""

    checkpoint_id: str
    parent_checkpoint_id: str | None
    created_at: str | None
    next_nodes: tuple[str, ...]
    source: str | None
    step: int | None
    interrupt_count: int
    task_count: int
    task_errors: tuple[str, ...]

    @classmethod
    def from_snapshot(cls, snapshot: StateSnapshot) -> CheckpointView:
        """Project a LangGraph state snapshot into stable metadata."""

        identifier = checkpoint_id(snapshot.config)
        if identifier is None:
            raise ValueError("LangGraph snapshot is missing configurable.checkpoint_id")
        metadata = snapshot.metadata or {}
        source_value = metadata.get("source")
        step_value = metadata.get("step")
        errors = tuple(str(task.error) for task in snapshot.tasks if task.error is not None)
        return cls(
            checkpoint_id=identifier,
            parent_checkpoint_id=checkpoint_id(snapshot.parent_config),
            created_at=snapshot.created_at,
            next_nodes=tuple(snapshot.next),
            source=str(source_value) if source_value is not None else None,
            step=int(step_value) if isinstance(step_value, int) else None,
            interrupt_count=len(snapshot.interrupts),
            task_count=len(snapshot.tasks),
            task_errors=errors,
        )


@dataclass(frozen=True, slots=True)
class ForkResult:
    """A branch checkpoint created by ``update_state``."""

    config: RunnableConfig
    checkpoint_id: str
    parent_checkpoint_id: str


class StatefulGraph(Protocol):
    """Public CompiledStateGraph surface used by Caro."""

    def get_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot: ...

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]: ...

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig: ...

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any: ...

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: str | Sequence[str] | None = None,
        version: str = "v1",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any] | Any]: ...


class CheckpointDatabase:
    """Own the SQLite connection required by LangGraph's synchronous saver."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        self.saver.setup()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self.connection.close()

    def __enter__(self) -> CheckpointDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class GraphController:
    """Expose recoverability and time travel without altering graph execution."""

    def __init__(self, graph: StatefulGraph) -> None:
        self.graph = graph

    def state(self, thread_id: str, *, checkpoint: str | None = None) -> StateSnapshot:
        """Read the latest or a specific thread snapshot."""

        return self.graph.get_state(graph_config(thread_id, checkpoint_id=checkpoint))

    def history(self, thread_id: str, *, limit: int = 50) -> list[CheckpointView]:
        """Read reverse-chronological checkpoint metadata."""

        snapshots = self.graph.get_state_history(graph_config(thread_id), limit=limit)
        return [CheckpointView.from_snapshot(snapshot) for snapshot in snapshots]

    def resolve_snapshot(self, thread_id: str, checkpoint: str) -> StateSnapshot:
        """Resolve a concrete checkpoint and fail rather than silently using latest."""

        # Use history rather than a minimal get_state lookup: SQLiteSaver's
        # update_state path requires the complete checkpoint config, including
        # checkpoint_ns, which history snapshots preserve.
        for snapshot in self.graph.get_state_history(graph_config(thread_id)):
            if checkpoint_id(snapshot.config) == checkpoint:
                return snapshot
        raise ValueError(f"checkpoint not found in thread {thread_id}: {checkpoint}")

    def fork(
        self,
        thread_id: str,
        checkpoint: str,
        values: Mapping[str, Any],
        *,
        as_node: str | None = None,
    ) -> ForkResult:
        """Create a same-thread branch from a past checkpoint using ``update_state``."""

        snapshot = self.resolve_snapshot(thread_id, checkpoint)
        updated = self.graph.update_state(snapshot.config, dict(values), as_node=as_node)
        new_identifier = checkpoint_id(updated)
        if new_identifier is None:
            raise ValueError("LangGraph update_state returned no checkpoint_id")
        return ForkResult(config=updated, checkpoint_id=new_identifier, parent_checkpoint_id=checkpoint)

    def replay(self, thread_id: str, checkpoint: str) -> dict[str, Any] | Any:
        """Replay graph execution after a checkpoint; downstream nodes run again."""

        snapshot = self.resolve_snapshot(thread_id, checkpoint)
        return self.graph.invoke(None, snapshot.config)


def coerce_state_mapping(snapshot: StateSnapshot) -> dict[str, Any]:
    """Return mapping state for CLI inspection and reject non-mapping graph state."""

    if not isinstance(snapshot.values, Mapping):
        raise TypeError("Caro requires mapping-shaped LangGraph state")
    return cast("dict[str, Any]", dict(snapshot.values))

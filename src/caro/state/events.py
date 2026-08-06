"""Normalize LangGraph v2 stream chunks for the terminal renderer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from caro.state.checkpoints import StatefulGraph


class GraphEventKind(StrEnum):
    """Stable event categories used by Caro's CLI."""

    MESSAGE = "message"
    UPDATE = "update"
    TASK = "task"
    CHECKPOINT = "checkpoint"
    INTERRUPT = "interrupt"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GraphEvent:
    """One normalized low-level LangGraph stream event."""

    kind: GraphEventKind
    data: Any
    node: str | None = None
    text: str | None = None
    config: RunnableConfig | None = None


def normalize_chunk(chunk: Any) -> list[GraphEvent]:
    """Normalize LangGraph v2 chunks while tolerating custom/unknown payloads."""

    if not isinstance(chunk, dict) or "type" not in chunk:
        return [GraphEvent(GraphEventKind.UNKNOWN, chunk)]
    event_type = chunk.get("type")
    data = chunk.get("data")
    if event_type == "messages":
        message: Any
        metadata: Any
        if isinstance(data, tuple) and len(data) == 2:
            message, metadata = data
        else:
            message, metadata = data, {}
        node_value = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
        text = message.text if isinstance(message, BaseMessage) else None
        return [GraphEvent(GraphEventKind.MESSAGE, data, node=str(node_value) if node_value else None, text=text)]
    if event_type == "updates":
        if isinstance(data, dict) and "__interrupt__" in data:
            return [GraphEvent(GraphEventKind.INTERRUPT, data["__interrupt__"])]
        if isinstance(data, dict):
            return [GraphEvent(GraphEventKind.UPDATE, value, node=str(node)) for node, value in data.items()]
        return [GraphEvent(GraphEventKind.UPDATE, data)]
    if event_type == "tasks":
        return [GraphEvent(GraphEventKind.TASK, data)]
    if event_type == "checkpoints":
        checkpoint_config = data.get("config") if isinstance(data, dict) else None
        return [
            GraphEvent(
                GraphEventKind.CHECKPOINT,
                data,
                config=(
                    cast("RunnableConfig", checkpoint_config)
                    if isinstance(checkpoint_config, dict)
                    else None
                ),
            )
        ]
    return [GraphEvent(GraphEventKind.UNKNOWN, chunk)]


def stream_graph_events(
    graph: StatefulGraph,
    graph_input: Any,
    config: RunnableConfig,
    *,
    modes: Sequence[str] = ("messages", "updates"),
    capture_checkpoints: bool = False,
) -> Iterator[GraphEvent]:
    """Stream typed events directly from the existing compiled DeepAgents graph."""

    stream_modes = list(modes)
    if capture_checkpoints and "checkpoints" not in stream_modes:
        stream_modes.append("checkpoints")
    for chunk in graph.stream(graph_input, config, stream_mode=stream_modes, version="v2"):
        yield from normalize_chunk(chunk)

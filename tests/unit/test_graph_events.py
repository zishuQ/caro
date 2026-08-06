"""LangGraph v2 event normalization tests."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk

from caro.state.events import GraphEventKind, normalize_chunk, stream_graph_events


def test_message_chunk_exposes_text_and_node() -> None:
    events = normalize_chunk(
        {
            "type": "messages",
            "data": (AIMessageChunk(content="hello"), {"langgraph_node": "model"}),
        }
    )
    assert events[0].kind is GraphEventKind.MESSAGE
    assert events[0].text == "hello"
    assert events[0].node == "model"


def test_updates_are_split_by_node() -> None:
    events = normalize_chunk({"type": "updates", "data": {"one": {"x": 1}, "two": {"y": 2}}})
    assert [(event.kind, event.node) for event in events] == [
        (GraphEventKind.UPDATE, "one"),
        (GraphEventKind.UPDATE, "two"),
    ]


def test_interrupt_and_unknown_are_preserved() -> None:
    interrupt = normalize_chunk({"type": "updates", "data": {"__interrupt__": ["approval"]}})
    unknown = normalize_chunk({"type": "custom", "data": {"x": 1}})
    assert interrupt[0].kind is GraphEventKind.INTERRUPT
    assert unknown[0].kind is GraphEventKind.UNKNOWN


def test_checkpoint_chunk_preserves_branch_config() -> None:
    config = {"configurable": {"thread_id": "thread", "checkpoint_id": "cp-1"}}
    events = normalize_chunk(
        {"type": "checkpoints", "data": {"config": config, "values": {"x": 1}}}
    )
    assert events[0].kind is GraphEventKind.CHECKPOINT
    assert events[0].config == config


def test_stream_capture_requests_checkpoint_mode() -> None:
    captured: dict[str, Any] = {}

    class Graph:
        def stream(self, _input: Any, _config: Any, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

    list(stream_graph_events(Graph(), {}, {}, modes=("updates",), capture_checkpoints=True))
    assert captured["stream_mode"] == ["updates", "checkpoints"]

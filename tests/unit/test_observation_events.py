"""Structured Langfuse event sink tests."""

from __future__ import annotations

from typing import Any

from caro.observability.events import (
    ApprovalObservationEvent,
    LangfuseObservationEventSink,
    RunObservationEvent,
)


class FakeClient:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.flushed = False

    def create_event(self, **kwargs: object) -> object:
        self.events.append(dict(kwargs))
        return object()

    def flush(self) -> None:
        self.flushed = True


def test_events_are_branded_caro_and_redacted() -> None:
    client = FakeClient()
    sink = LangfuseObservationEventSink(client)

    sink.record_approval(
        ApprovalObservationEvent(
            thread_id="thread",
            run_id="run",
            tool_name="execute",
            decision="approved",
            mode="terminal",
            metadata={"api_key": "secret-value"},
        )
    )
    sink.record_run(RunObservationEvent(thread_id="thread", run_id="run", phase="completed"))
    sink.flush()

    assert [event["name"] for event in client.events] == ["caro.approval", "caro.run"]
    assert client.events[0]["metadata"]["metadata"]["api_key"] == "[REDACTED]"
    assert client.flushed is True

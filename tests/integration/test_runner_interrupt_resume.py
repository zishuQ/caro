"""CaroRunner HITL pause and checkpoint resume integration test."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from caro.config import CaroSettings
from caro.observability.events import ApprovalObservationEvent, RunObservationEvent
from caro.observability.langfuse import TraceContext
from caro.runner import CaroRunner
from caro.state.checkpoints import CheckpointDatabase
from caro.state.events import GraphEvent
from caro.state.runs import RunStatus, RunStore


class ApprovalState(TypedDict, total=False):
    messages: list[dict[str, str]]
    decision: str


def approval_node(_: ApprovalState) -> ApprovalState:
    response = interrupt(
        {
            "action_requests": [{"name": "write_file", "args": {"path": "README.md"}}],
            "review_configs": [{"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}],
        }
    )
    return {"decision": response["decisions"][0]["type"]}


def build_approval_graph(database: CheckpointDatabase):
    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=database.saver)


def no_trace(_: CaroSettings, __: Any) -> TraceContext:
    return TraceContext()


def test_runner_pauses_then_resumes_same_checkpoint(tmp_path: Path) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
        graph={"stream_modes": ["updates", "checkpoints"]},
    )
    store = RunStore(tmp_path / "runs.sqlite3")
    events: list[GraphEvent] = []
    with CheckpointDatabase(tmp_path / "checkpoints.sqlite3") as database:
        graph = build_approval_graph(database)
        paused = CaroRunner(graph, store, settings, event_sink=events.append, trace_factory=no_trace).run("edit readme")
        assert paused.status is RunStatus.WAITING_APPROVAL
        assert paused.checkpoint_id is not None

        def approve(request: dict[str, Any]) -> list[dict[str, Any]]:
            assert request["action_requests"][0]["name"] == "write_file"
            return [{"type": "approve"}]

        completed = CaroRunner(
            graph,
            store,
            settings,
            approval_handler=approve,
            event_sink=events.append,
            trace_factory=no_trace,
        ).resume(paused.run_id)

        assert completed.status is RunStatus.COMPLETED
        assert graph.get_state({"configurable": {"thread_id": completed.thread_id}}).values["decision"] == "approve"
        assert any(event.kind.value == "interrupt" for event in events)


def test_runner_always_propagates_run_provenance_without_langfuse(tmp_path: Path) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
        graph={"stream_modes": ["updates"]},
    )
    store = RunStore(tmp_path / "runs.sqlite3")
    observed: dict[str, Any] = {}

    def observe_metadata(state: ApprovalState, config: RunnableConfig) -> ApprovalState:
        del state
        observed.update(config.get("metadata", {}))
        return {"decision": "done"}

    with CheckpointDatabase(tmp_path / "checkpoints.sqlite3") as database:
        builder = StateGraph(ApprovalState)
        builder.add_node("observe", observe_metadata)
        builder.add_edge(START, "observe")
        builder.add_edge("observe", END)
        graph = builder.compile(checkpointer=database.saver)
        completed = CaroRunner(graph, store, settings, trace_factory=no_trace).run("record provenance")

    assert completed.status is RunStatus.COMPLETED
    assert observed["caro_run_id"] == completed.run_id
    assert observed["caro_thread_id"] == completed.thread_id
    assert observed["caro_memory_enabled"] is True


def test_runner_records_hitl_and_run_observations(tmp_path: Path) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
        graph={"stream_modes": ["updates"]},
    )
    store = RunStore(tmp_path / "runs.sqlite3")
    approvals: list[ApprovalObservationEvent] = []
    runs: list[RunObservationEvent] = []

    class Sink:
        def record_approval(self, event: ApprovalObservationEvent) -> None:
            approvals.append(event)

        def record_run(self, event: RunObservationEvent) -> None:
            runs.append(event)

        def flush(self) -> None:
            return

    sink = Sink()

    def trace(_: CaroSettings, __: Any) -> TraceContext:
        return TraceContext(observation_sink=sink)

    with CheckpointDatabase(tmp_path / "checkpoints.sqlite3") as database:
        graph = build_approval_graph(database)
        paused = CaroRunner(graph, store, settings, trace_factory=trace).run("edit readme")

        def approve(_: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"type": "approve"}]

        CaroRunner(
            graph,
            store,
            settings,
            approval_handler=approve,
            trace_factory=trace,
        ).resume(paused.run_id)

    assert [event.decision for event in approvals] == ["requested", "requested", "approved"]
    assert runs[0].phase == "running"
    assert runs[-1].phase == "completed"

"""Drive a compiled LangGraph in the terminal without reimplementing its loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StateSnapshot

from caro.config import CaroSettings
from caro.observability.events import ApprovalDecision, ApprovalObservationEvent, RunObservationEvent
from caro.observability.langfuse import TraceContext, build_trace_context
from caro.state.checkpoints import GraphController, StatefulGraph, checkpoint_id, graph_config
from caro.state.events import GraphEvent, stream_graph_events
from caro.state.runs import RunRecord, RunStatus, RunStore
from caro.tools.registry import ToolRegistry

Decision = dict[str, Any]
ApprovalHandler = Callable[[dict[str, Any]], list[Decision]]
EventSink = Callable[[GraphEvent], None]
TraceFactory = Callable[[CaroSettings, RunRecord], TraceContext]
logger = logging.getLogger(__name__)


class InterruptLike(Protocol):
    """Subset of LangGraph Interrupt used by the runner."""

    id: str
    value: Any


def _resume_command(
    snapshot: StateSnapshot,
    approval_handler: ApprovalHandler,
    *,
    collected: list[Decision] | None = None,
) -> Command[Any]:
    """Build single- or multi-interrupt resume input from terminal decisions."""

    interrupts = cast("Sequence[InterruptLike]", snapshot.interrupts)
    if not interrupts:
        raise ValueError("cannot resume approval: checkpoint has no interrupts")
    responses: list[tuple[str, dict[str, Any]]] = []
    for pending in interrupts:
        if not isinstance(pending.value, dict):
            raise TypeError("Caro only supports mapping-shaped approval interrupts")
        decisions = approval_handler(pending.value)
        actions = pending.value.get("action_requests", [])
        if isinstance(actions, list):
            decisions = _sanitize_approval_decisions(actions, decisions)
        if collected is not None:
            collected.extend(decisions)
        responses.append((pending.id, {"decisions": decisions}))
    if len(responses) == 1:
        return Command(resume=responses[0][1])
    return Command(resume={identifier: response for identifier, response in responses})


def _sanitize_approval_decisions(
    actions: list[Any], decisions: list[Decision]
) -> list[Decision]:
    """Prevent an edited memory approval from reintroducing sensitive data."""

    sanitized: list[Decision] = []
    for index, decision in enumerate(decisions):
        if (
            index < len(actions)
            and isinstance(actions[index], dict)
            and actions[index].get("name") == "memory_propose"
            and decision.get("type") == "edit"
        ):
            sanitized.append(
                {
                    "type": "reject",
                    "message": "Caro does not allow edited memory proposals; submit a new verified proposal.",
                }
            )
        else:
            sanitized.append(decision)
    return sanitized


def _execution_config(
    config: RunnableConfig,
    trace: TraceContext,
    run: RunRecord,
    settings: CaroSettings,
) -> RunnableConfig:
    """Attach durable Caro provenance even when external tracing is disabled."""

    merged = cast("RunnableConfig", dict(config))
    if trace.callbacks:
        merged["callbacks"] = trace.callbacks
    metadata = dict(config.get("metadata", {}))
    metadata.update(
        {
            "caro_run_id": run.run_id,
            "caro_thread_id": run.thread_id,
            "caro_config_hash": run.config_hash,
            "caro_memory_enabled": settings.memory.enabled and settings.memory.structured_enabled,
        }
    )
    metadata.update(trace.metadata)
    merged["metadata"] = metadata
    return merged


class CaroRunner:
    """Persist lifecycle state while LangGraph owns execution and recovery."""

    def __init__(
        self,
        graph: StatefulGraph,
        store: RunStore,
        settings: CaroSettings,
        *,
        approval_handler: ApprovalHandler | None = None,
        event_sink: EventSink | None = None,
        trace_factory: TraceFactory = build_trace_context,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.settings = settings
        self.approval_handler = approval_handler
        self.event_sink = event_sink
        self.trace_factory = trace_factory
        self.registry = registry
        self.controller = GraphController(graph)

    def _validate_run_context(self, run: RunRecord) -> None:
        """Reject recovery against a different workspace or graph configuration."""

        workspace = self.settings.workspace.root.expanduser().resolve()
        recorded_workspace = str(workspace)
        if run.workspace != recorded_workspace:
            raise ValueError(
                f"run workspace mismatch: recorded {run.workspace!r}, current {recorded_workspace!r}"
            )
        current_hash = self.settings.fingerprint()
        if run.config_hash != current_hash:
            raise ValueError(
                "run configuration fingerprint does not match the current settings; "
                "use an explicit migration or start a new run"
            )

    def run(self, task: str) -> RunRecord:
        """Create and execute a new task thread."""

        run = self.store.create(
            task=task,
            workspace=self.settings.workspace.root,
            config_hash=self.settings.fingerprint(),
        )
        config = graph_config(run.thread_id, recursion_limit=self.settings.graph.recursion_limit)
        graph_input = {"messages": [{"role": "user", "content": task}]}
        return self._execute(run, graph_input, config)

    def resume(self, run_id: str) -> RunRecord:
        """Resume a paused, failed, or approval-waiting run from its checkpoint."""

        run = self.store.get(run_id)
        self._validate_run_context(run)
        if run.status not in {
            RunStatus.CREATED,
            RunStatus.RUNNING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.WAITING_APPROVAL,
        }:
            raise ValueError(f"run cannot be resumed from status {run.status.value}")
        if run.checkpoint_id is not None:
            snapshot = self.controller.resolve_snapshot(run.thread_id, run.checkpoint_id)
        else:
            snapshot = self.controller.state(run.thread_id)
        graph_input: Any = None
        decisions: list[Decision] = []
        if snapshot.interrupts:
            if self.approval_handler is None:
                return run
            graph_input = _resume_command(snapshot, self.approval_handler, collected=decisions)
        return self._execute(
            run,
            graph_input,
            snapshot.config,
            initial_snapshot=snapshot if snapshot.interrupts else None,
            initial_decisions=decisions,
        )

    def replay(self, parent_run_id: str, checkpoint: str) -> RunRecord:
        """Create an audited child run and replay downstream nodes from a checkpoint."""

        parent = self.store.get(parent_run_id)
        self._validate_run_context(parent)
        snapshot = self.controller.resolve_snapshot(parent.thread_id, checkpoint)
        child = self.store.create(
            task=parent.task,
            workspace=self.settings.workspace.root,
            config_hash=self.settings.fingerprint(),
            thread_id=parent.thread_id,
            parent_run_id=parent.run_id,
            checkpoint_id=checkpoint,
            status=RunStatus.PAUSED,
        )
        return self._execute(child, None, snapshot.config)

    def retry(self, run_id: str) -> RunRecord:
        """Retry a failed run as a new child with explicit lineage."""

        parent = self.store.get(run_id)
        self._validate_run_context(parent)
        if parent.status is not RunStatus.FAILED:
            raise ValueError(f"only failed runs can be retried: {parent.status.value}")
        if parent.checkpoint_id is not None:
            snapshot = self.controller.resolve_snapshot(parent.thread_id, parent.checkpoint_id)
            child = self.store.create(
                task=parent.task,
                workspace=self.settings.workspace.root,
                config_hash=self.settings.fingerprint(),
                thread_id=parent.thread_id,
                parent_run_id=parent.run_id,
                checkpoint_id=parent.checkpoint_id,
                status=RunStatus.PAUSED,
            )
            return self._execute(child, None, snapshot.config)
        child = self.store.create(
            task=parent.task,
            workspace=self.settings.workspace.root,
            config_hash=self.settings.fingerprint(),
            parent_run_id=parent.run_id,
        )
        config = graph_config(child.thread_id, recursion_limit=self.settings.graph.recursion_limit)
        return self._execute(child, {"messages": [{"role": "user", "content": parent.task}]}, config)

    def cancel(self, run_id: str) -> RunRecord:
        """Cancel a not-yet-terminal run in the durable lifecycle store."""

        run = self.store.get(run_id)
        if run.status in {RunStatus.COMPLETED, RunStatus.CANCELLED}:
            raise ValueError(f"run cannot be cancelled from status {run.status.value}")
        return self.store.transition(
            run.run_id,
            RunStatus.CANCELLED,
            payload={"reason": "cancelled by Caro operator"},
        )

    def _record_run_observation(
        self,
        trace: TraceContext,
        run: RunRecord,
        phase: str,
        *,
        previous_phase: str | None = None,
        started_at: float | None = None,
        error: Exception | None = None,
    ) -> None:
        """Send a fail-open lifecycle event to the configured observation sink."""

        try:
            trace.observation_sink.record_run(
                RunObservationEvent(
                    thread_id=run.thread_id,
                    run_id=run.run_id,
                    phase=phase,
                    previous_phase=previous_phase,
                    latency_ms=(
                        max(0, int((time.perf_counter() - started_at) * 1000))
                        if started_at is not None
                        else None
                    ),
                    error_type=type(error).__name__ if error is not None else None,
                    error_message=str(error) if error is not None else None,
                )
            )
        except Exception as observation_error:  # Telemetry is fail-open.
            logger.debug("Could not record Caro run observation: %s", observation_error)

    def _record_approval_observations(
        self,
        trace: TraceContext,
        run: RunRecord,
        snapshot: StateSnapshot,
        decisions: list[Decision] | None,
        *,
        mode: str,
    ) -> None:
        """Record each LangGraph HITL action and terminal decision."""

        decision_values = decisions or []
        decision_index = 0
        for pending in cast("Sequence[InterruptLike]", snapshot.interrupts):
            if not isinstance(pending.value, dict):
                continue
            actions = pending.value.get("action_requests", [])
            if not isinstance(actions, list):
                continue
            for action in actions:
                if not isinstance(action, dict):
                    continue
                name = str(action.get("name", "unknown"))
                raw_args = action.get("args")
                args = raw_args if isinstance(raw_args, dict) else {}
                assessment = self.registry.assess(name, args) if self.registry is not None else None
                raw_decision = (
                    decision_values[decision_index].get("type")
                    if decision_index < len(decision_values)
                    else None
                )
                if decision_index < len(decision_values):
                    decision_index += 1
                decision_map: dict[str, ApprovalDecision] = {
                    "approve": "approved",
                    "edit": "approved",
                    "reject": "rejected",
                }
                decision = decision_map.get(str(raw_decision), "requested")
                try:
                    trace.observation_sink.record_approval(
                        ApprovalObservationEvent(
                            thread_id=run.thread_id,
                            run_id=run.run_id,
                            tool_name=name,
                            decision=decision,
                            mode=mode,
                            risk_level=assessment.risk.value if assessment is not None else None,
                            reason=assessment.reason if assessment is not None else None,
                            metadata=dict(assessment.metadata) if assessment is not None else {},
                        )
                    )
                except Exception as observation_error:  # Telemetry is fail-open.
                    logger.debug("Could not record Caro approval observation: %s", observation_error)

    def _execute(
        self,
        run: RunRecord,
        graph_input: Any,
        config: RunnableConfig,
        *,
        initial_snapshot: StateSnapshot | None = None,
        initial_decisions: list[Decision] | None = None,
    ) -> RunRecord:
        run = self.store.transition(run.run_id, RunStatus.RUNNING)
        trace: TraceContext | None = None
        started_at = time.perf_counter()
        try:
            trace = self.trace_factory(self.settings, run)
        except Exception as error:
            current = self.store.get(run.run_id)
            if current.status is RunStatus.RUNNING:
                self.store.transition(run.run_id, RunStatus.FAILED, error=str(error))
            raise
        assert trace is not None
        self._record_run_observation(trace, run, RunStatus.RUNNING.value)
        if initial_snapshot is not None:
            self._record_approval_observations(
                trace,
                run,
                initial_snapshot,
                None,
                mode="checkpoint",
            )
            self._record_approval_observations(
                trace,
                run,
                initial_snapshot,
                initial_decisions,
                mode="terminal",
            )
        latest_config = config
        try:
            while True:
                traced_config = _execution_config(config, trace, run, self.settings)
                for event in stream_graph_events(
                    self.graph,
                    graph_input,
                    traced_config,
                    modes=self.settings.graph.stream_modes,
                    capture_checkpoints=True,
                ):
                    if event.config is not None:
                        latest_config = event.config
                    if self.store.get(run.run_id).status is RunStatus.CANCELLED:
                        cancelled = self.store.get(run.run_id)
                        self._record_run_observation(
                            trace,
                            cancelled,
                            RunStatus.CANCELLED.value,
                            previous_phase=RunStatus.RUNNING.value,
                            started_at=started_at,
                        )
                        return cancelled
                    if self.event_sink is not None:
                        self.event_sink(event)

                current = self.store.get(run.run_id)
                if current.status is RunStatus.CANCELLED:
                    self._record_run_observation(
                        trace,
                        current,
                        RunStatus.CANCELLED.value,
                        previous_phase=RunStatus.RUNNING.value,
                        started_at=started_at,
                    )
                    return current
                snapshot = self.graph.get_state(latest_config)
                identifier = checkpoint_id(snapshot.config)
                if identifier is not None:
                    run = self.store.set_checkpoint(run.run_id, identifier)
                if snapshot.interrupts:
                    self._record_approval_observations(trace, run, snapshot, None, mode="checkpoint")
                    run = self.store.transition(run.run_id, RunStatus.WAITING_APPROVAL)
                    self._record_run_observation(
                        trace,
                        run,
                        RunStatus.WAITING_APPROVAL.value,
                        previous_phase=RunStatus.RUNNING.value,
                        started_at=started_at,
                    )
                    if self.approval_handler is None:
                        return run
                    decisions: list[Decision] = []
                    graph_input = _resume_command(snapshot, self.approval_handler, collected=decisions)
                    self._record_approval_observations(trace, run, snapshot, decisions, mode="terminal")
                    run = self.store.transition(run.run_id, RunStatus.RUNNING)
                    config = snapshot.config
                    latest_config = config
                    continue
                if snapshot.next:
                    paused = self.store.transition(run.run_id, RunStatus.PAUSED)
                    self._record_run_observation(
                        trace,
                        paused,
                        RunStatus.PAUSED.value,
                        previous_phase=RunStatus.RUNNING.value,
                        started_at=started_at,
                    )
                    return paused
                completed = self.store.transition(run.run_id, RunStatus.COMPLETED)
                self._record_run_observation(
                    trace,
                    completed,
                    RunStatus.COMPLETED.value,
                    previous_phase=RunStatus.RUNNING.value,
                    started_at=started_at,
                )
                return completed
        except Exception as error:
            if latest_config is not None:
                identifier = checkpoint_id(latest_config)
                if identifier is not None:
                    try:
                        run = self.store.set_checkpoint(run.run_id, identifier)
                    except Exception as checkpoint_error:
                        logger.debug("Could not persist failed-run checkpoint: %s", checkpoint_error)
            current = self.store.get(run.run_id)
            if current.status is RunStatus.CANCELLED:
                return current
            if current.status in {RunStatus.RUNNING, RunStatus.WAITING_APPROVAL}:
                failed = self.store.transition(run.run_id, RunStatus.FAILED, error=str(error))
                self._record_run_observation(
                    trace,
                    failed,
                    RunStatus.FAILED.value,
                    previous_phase=current.status.value,
                    started_at=started_at,
                    error=error,
                )
            raise
        finally:
            if trace is not None:
                trace.close()

"""Deterministic multi-agent orchestration using an outer LangGraph StateGraph."""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from caro.config import AgentRoleSettings, OrchestrationSettings
from caro.errors import CaroError
from caro.state.checkpoints import StatefulGraph


class AgentReport(TypedDict):
    """One durable role result written into shared orchestration state."""

    agent: str
    stage: str
    stage_index: int
    model: str
    content: str


class OrchestrationState(MessagesState):
    """Shared state for deterministic stage fan-out and fan-in."""

    task: str
    reports: Annotated[list[AgentReport], operator.add]


class RoleRunnable(Protocol):
    """Minimal compiled DeepAgent surface invoked by a role node."""

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any: ...


def _tool_name(tool: Any) -> str | None:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if not isinstance(tool, Mapping):
        return None
    direct = tool.get("name")
    if isinstance(direct, str):
        return direct
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return str(function["name"])
    return None


class NestedDelegationGuardMiddleware(AgentMiddleware[Any, Any, Any]):
    """Hide and reject nested `task` calls inside deterministic role nodes."""

    @staticmethod
    def _without_task(request: ModelRequest[Any]) -> ModelRequest[Any]:
        return request.override(tools=[tool for tool in request.tools if _tool_name(tool) != "task"])

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        return handler(self._without_task(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        return await handler(self._without_task(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call.get("name") == "task":
            raise CaroError("nested task delegation is disabled by Caro orchestration policy")
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call.get("name") == "task":
            raise CaroError("nested task delegation is disabled by Caro orchestration policy")
        return await handler(request)


def _message_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return message.text
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _original_task(state: OrchestrationState) -> str:
    existing = state.get("task")
    if existing:
        return existing
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.text
        if isinstance(message, Mapping) and message.get("role") == "user":
            text = _message_text(message)
            if text:
                return text
    raise CaroError("orchestration input must contain a user task")


def _agent_output(result: Any) -> str:
    if not isinstance(result, Mapping):
        raise CaroError("an orchestrated agent returned non-mapping state")
    messages = result.get("messages")
    if not isinstance(messages, list):
        raise CaroError("an orchestrated agent returned no messages")
    for message in reversed(messages):
        if isinstance(message, AIMessage) or (isinstance(message, Mapping) and message.get("role") == "assistant"):
            text = _message_text(message).strip()
            if text:
                return text
    raise CaroError("an orchestrated agent returned no final assistant report")


def _bounded_report(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    marker = "\n\n[Report truncated by Caro orchestration]"
    return content[: max(0, limit - len(marker))].rstrip() + marker


def _handoff_context(reports: list[AgentReport], limit: int) -> str:
    if not reports:
        return "No earlier-stage reports are available."
    sections = [
        f"### Stage {report['stage']} · Agent {report['agent']}\n\n{report['content']}"
        for report in sorted(reports, key=lambda item: (item["stage_index"], item["agent"]))
    ]
    content = "\n\n---\n\n".join(sections)
    if len(content) <= limit:
        return content
    marker = "[Earlier report context truncated by Caro]\n\n"
    return marker + content[-max(0, limit - len(marker)) :]


@dataclass(frozen=True, slots=True)
class RoleExecutor:
    """Map shared parent state into a private DeepAgent invocation and back."""

    agent_name: str
    stage_name: str
    stage_index: int
    model_name: str
    role: AgentRoleSettings
    runnable: RoleRunnable
    max_report_chars: int
    max_handoff_chars: int

    def __call__(
        self,
        state: OrchestrationState,
        config: RunnableConfig,
    ) -> dict[str, list[AgentReport]]:
        task = _original_task(state)
        prior_reports = [report for report in state.get("reports", []) if report["stage_index"] < self.stage_index]
        handoff = _handoff_context(prior_reports, self.max_handoff_chars)
        prompt = (
            f"Original coding task:\n\n{task}\n\n"
            f"Reports completed by earlier orchestration stages:\n\n{handoff}\n\n"
            f"Your assigned role is `{self.agent_name}`: {self.role.description}\n\n"
            "Perform only this role's part of the task. Use workspace tools when needed. "
            "Return a self-contained report for later agents and the final operator."
        )
        result = self.runnable.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        content = _bounded_report(_agent_output(result), self.max_report_chars)
        report: AgentReport = {
            "agent": self.agent_name,
            "stage": self.stage_name,
            "stage_index": self.stage_index,
            "model": self.model_name,
            "content": content,
        }
        return {"reports": [report]}


def _initialize(state: OrchestrationState) -> dict[str, str]:
    return {"task": _original_task(state)}


def _barrier(state: OrchestrationState) -> dict[str, str]:
    """Create an explicit fan-in checkpoint after every stage."""

    return {"task": state["task"]}


def _finalize(state: OrchestrationState) -> dict[str, list[AIMessage]]:
    reports = state.get("reports", [])
    if not reports:
        raise CaroError("orchestration completed without agent reports")
    final_stage = max(report["stage_index"] for report in reports)
    final_reports = sorted(
        (report for report in reports if report["stage_index"] == final_stage),
        key=lambda item: item["agent"],
    )
    if len(final_reports) == 1:
        content = final_reports[0]["content"]
    else:
        content = "\n\n---\n\n".join(
            f"## {report['agent']}\n\n{report['content']}" for report in final_reports
        )
    return {"messages": [AIMessage(content=content, name="caro-orchestration")]}


def compile_orchestration(
    settings: OrchestrationSettings,
    role_graphs: Mapping[str, RoleRunnable],
    role_models: Mapping[str, str],
    checkpointer: BaseCheckpointSaver[Any],
) -> StatefulGraph:
    """Compile configured sequential stages with parallel role nodes."""

    if not settings.enabled:
        raise ValueError("cannot compile disabled orchestration")
    missing_graphs = sorted(set(settings.agents).difference(role_graphs))
    if missing_graphs:
        raise ValueError(f"missing compiled role graph(s): {', '.join(missing_graphs)}")

    builder = StateGraph(OrchestrationState)
    builder.add_node("orchestration_start", _initialize)
    builder.add_edge(START, "orchestration_start")
    previous: str = "orchestration_start"

    for stage_index, stage in enumerate(settings.stages):
        stage_nodes: list[str] = []
        for agent_name in stage.agents:
            node_name = f"stage_{stage_index:02d}__{agent_name.replace('-', '_')}"
            stage_nodes.append(node_name)
            builder.add_node(
                node_name,
                RoleExecutor(
                    agent_name=agent_name,
                    stage_name=stage.name,
                    stage_index=stage_index,
                    model_name=role_models[agent_name],
                    role=settings.agents[agent_name],
                    runnable=role_graphs[agent_name],
                    max_report_chars=settings.max_report_chars,
                    max_handoff_chars=settings.max_handoff_chars,
                ),
            )
            builder.add_edge(previous, node_name)

        barrier = f"stage_{stage_index:02d}__complete"
        builder.add_node(barrier, _barrier)
        builder.add_edge(stage_nodes if len(stage_nodes) > 1 else stage_nodes[0], barrier)
        previous = barrier

    builder.add_node("orchestration_finalize", _finalize)
    builder.add_edge(previous, "orchestration_finalize")
    builder.add_edge("orchestration_finalize", END)
    return cast(
        "StatefulGraph",
        builder.compile(checkpointer=checkpointer, name="caro_orchestration"),
    )

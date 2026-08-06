"""Deterministic LangGraph multi-agent orchestration tests."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from caro.config import OrchestrationSettings
from caro.orchestration import NestedDelegationGuardMiddleware, compile_orchestration
from caro.state.checkpoints import graph_config


class FakeRole:
    """Record handoff prompts and return one deterministic report."""

    def __init__(self, name: str, prompts: dict[str, str], barrier: threading.Barrier | None = None) -> None:
        self.name = name
        self.prompts = prompts
        self.barrier = barrier

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> dict[str, list[AIMessage]]:
        del config, kwargs
        assert isinstance(input, Mapping)
        messages = input["messages"]
        prompt = messages[-1].text
        self.prompts[self.name] = prompt
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        return {"messages": [AIMessage(content=f"{self.name} report")]}


def orchestration_settings() -> OrchestrationSettings:
    return OrchestrationSettings(
        enabled=True,
        agents={
            name: {
                "description": f"{name} role",
                "system_prompt": f"Act as {name}.",
            }
            for name in ("researcher", "architect", "implementer", "synthesizer")
        },
        stages=[
            {"name": "analysis", "agents": ["researcher", "architect"]},
            {"name": "implementation", "agents": ["implementer"]},
            {"name": "synthesis", "agents": ["synthesizer"]},
        ],
    )


def test_parallel_fanout_handoff_fanin_and_final_message() -> None:
    settings = orchestration_settings()
    prompts: dict[str, str] = {}
    parallel_barrier = threading.Barrier(2)
    roles = {
        name: FakeRole(name, prompts, parallel_barrier if name in {"researcher", "architect"} else None)
        for name in settings.agents
    }
    models = {name: "primary" for name in settings.agents}
    graph = compile_orchestration(settings, roles, models, InMemorySaver())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Fix the parser and verify it.")]},
        graph_config("orchestration-thread"),
    )

    assert len(result["reports"]) == 4
    assert result["messages"][-1].text == "synthesizer report"
    assert "No earlier-stage reports" in prompts["researcher"]
    assert "No earlier-stage reports" in prompts["architect"]
    assert "researcher report" in prompts["implementer"]
    assert "architect report" in prompts["implementer"]
    assert "implementer report" in prompts["synthesizer"]
    assert list(graph.get_state_history(graph_config("orchestration-thread")))


def test_parallel_final_stage_is_merged_deterministically() -> None:
    settings = OrchestrationSettings(
        enabled=True,
        agents={
            "reviewer": {"description": "Review", "system_prompt": "Review the diff."},
            "tester": {"description": "Test", "system_prompt": "Run tests."},
        },
        stages=[{"name": "verification", "agents": ["tester", "reviewer"]}],
    )
    prompts: dict[str, str] = {}
    barrier = threading.Barrier(2)
    graph = compile_orchestration(
        settings,
        {name: FakeRole(name, prompts, barrier) for name in settings.agents},
        {name: "primary" for name in settings.agents},
        InMemorySaver(),
    )

    result = graph.invoke({"messages": [HumanMessage(content="Verify changes.")]}, graph_config("merge-thread"))
    final = result["messages"][-1].text

    assert final.index("## reviewer") < final.index("## tester")
    assert "reviewer report" in final
    assert "tester report" in final


def test_nested_delegation_guard_removes_task_from_role_model_request() -> None:
    middleware = NestedDelegationGuardMiddleware()
    captured: list[str | None] = []
    request: ModelRequest[Any] = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="work")],
        tools=[{"name": "task"}, {"name": "read_file"}],
    )

    def handler(filtered: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.extend(tool.get("name") if isinstance(tool, dict) else tool.name for tool in filtered.tools)
        return ModelResponse(result=[AIMessage(content="ok")])

    result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, ModelResponse)
    assert captured == ["read_file"]


def test_parent_run_config_is_forwarded_to_role_subgraph() -> None:
    settings = OrchestrationSettings(
        enabled=True,
        agents={"reviewer": {"description": "Review", "system_prompt": "Review."}},
        stages=[{"name": "review", "agents": ["reviewer"]}],
    )
    observed: dict[str, Any] = {}

    class ConfigRole:
        def invoke(
            self,
            input: Any,
            config: RunnableConfig | None = None,
            **kwargs: Any,
        ) -> dict[str, list[AIMessage]]:
            del input, kwargs
            assert config is not None
            observed.update(config.get("metadata", {}))
            return {"messages": [AIMessage(content="reviewed")]}

    graph = compile_orchestration(
        settings,
        {"reviewer": ConfigRole()},
        {"reviewer": "primary"},
        InMemorySaver(),
    )
    config = graph_config("config-thread")
    config["metadata"] = {"caro_run_id": "run-orchestration-1"}

    graph.invoke({"messages": [HumanMessage(content="Review it.")]}, config)

    assert observed["caro_run_id"] == "run-orchestration-1"

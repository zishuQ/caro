"""DeepAgents public-extension assembly tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from caro.agent import assemble_agent
from caro.config import CaroSettings
from caro.memory.middleware import StructuredMemoryMiddleware
from caro.orchestration import NestedDelegationGuardMiddleware


class FakeGraph:
    pass


class BindableFakeMessagesModel(FakeMessagesListChatModel):
    """Cycle scripted messages while advertising the tool-binding API."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BindableFakeMessagesModel:
        """Accept tool schemas and keep the scripted model unchanged."""

        del tools, kwargs
        return self


def test_assembly_injects_governance_without_modifying_agent_loop(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_agent_factory(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    settings = CaroSettings(workspace={"root": tmp_path})
    assembly = assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=lambda _: FakeListChatModel(responses=["ok"]),
        agent_factory=fake_agent_factory,
    )

    assert assembly.graph.__class__ is FakeGraph
    assert captured["name"] == "caro"
    assert captured["checkpointer"].__class__ is InMemorySaver
    assert captured["backend"].cwd == tmp_path.resolve()
    assert captured["middleware"]
    assert captured["permissions"] == []
    assert "execute" in captured["interrupt_on"]
    assert "write_file" in captured["interrupt_on"]
    assert "memory_propose" in captured["interrupt_on"]
    assert "memory_archive" in captured["interrupt_on"]
    assert captured["memory"] == ["/.caro/memory/AGENTS.md"]
    assert {tool.name for tool in captured["tools"]} == {
        "memory_search",
        "memory_propose",
        "memory_archive",
    }
    assert any(isinstance(item, StructuredMemoryMiddleware) for item in captured["middleware"])
    assert "Project memory is scoped to this code workspace" in captured["system_prompt"]
    assert assembly.memory.statuses()[0].exists is True


def test_shell_can_be_disabled_while_files_stay_workspace_scoped(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_agent_factory(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    settings = CaroSettings(workspace={"root": tmp_path, "shell_enabled": False})
    assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=lambda _: FakeListChatModel(responses=["ok"]),
        agent_factory=fake_agent_factory,
    )
    assert captured["backend"].virtual_mode is True
    assert not hasattr(captured["backend"], "execute")
    assert captured["permissions"][0].mode == "interrupt"


def test_memory_can_be_disabled_without_changing_deepagents_loop(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_agent_factory(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    settings = CaroSettings(workspace={"root": tmp_path}, memory={"enabled": False})
    assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=lambda _: FakeListChatModel(responses=["ok"]),
        agent_factory=fake_agent_factory,
    )

    assert captured["memory"] is None
    assert captured["tools"] == []
    assert not (tmp_path / ".caro" / "memory" / "AGENTS.md").exists()


def test_orchestration_compiles_one_deepagent_per_role(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def fake_agent_factory(**kwargs: Any) -> FakeGraph:
        captured.append(kwargs)
        return FakeGraph()

    settings = CaroSettings(
        workspace={"root": tmp_path},
        orchestration={
            "enabled": True,
            "agents": {
                "reviewer": {
                    "description": "Review changes",
                    "system_prompt": "Review correctness.",
                    "shell_enabled": False,
                },
                "tester": {
                    "description": "Run tests",
                    "system_prompt": "Run focused tests.",
                },
            },
            "stages": [
                {"name": "review", "agents": ["reviewer"]},
                {"name": "test", "agents": ["tester"]},
            ],
        },
    )
    assembly = assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=lambda _: FakeListChatModel(responses=["ok"]),
        agent_factory=fake_agent_factory,
    )

    assert assembly.orchestrated is True
    assert set(assembly.role_graphs) == {"reviewer", "tester"}
    assert [item["name"] for item in captured] == ["caro_reviewer", "caro_tester"]
    assert all(item["checkpointer"] is None for item in captured)
    assert all(isinstance(item["middleware"][0], NestedDelegationGuardMiddleware) for item in captured)
    assert all(
        {tool.name for tool in item["tools"]} == {"memory_search", "memory_propose", "memory_archive"}
        for item in captured
    )
    assert captured[0]["permissions"]
    assert captured[1]["permissions"] == []


def test_orchestration_wires_allowlisted_delegate_models_and_governance(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    def fake_model_factory(definition: Any) -> FakeListChatModel:
        model = FakeListChatModel(responses=[definition.model])
        models[definition.model] = model
        return model

    def fake_agent_factory(**kwargs: Any) -> FakeGraph:
        captured.append(kwargs)
        return FakeGraph()

    settings = CaroSettings(
        workspace={"root": tmp_path, "shell_enabled": False},
        memory={"enabled": False},
        routing={
            "default": "high",
            "models": {
                "high": {"model": "openai:gpt-5.6-sol"},
                "cheap": {"model": "deepseek:deepseek-v4-flash"},
            },
        },
        orchestration={
            "enabled": True,
            "allow_nested_delegation": True,
            "agents": {
                "architect": {
                    "description": "Design the change.",
                    "system_prompt": "Analyze the architecture.",
                    "model": "high",
                    "delegates": [
                        {
                            "name": "cheap_researcher",
                            "description": "Gather focused evidence.",
                            "system_prompt": "Return evidence only.",
                            "model": "cheap",
                        }
                    ],
                }
            },
            "stages": [{"name": "analysis", "agents": ["architect"]}],
        },
    )
    assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=fake_model_factory,
        agent_factory=fake_agent_factory,
    )

    assert len(captured) == 1
    parent = captured[0]
    subagents = {spec["name"]: spec for spec in parent["subagents"]}
    assert set(subagents) == {"general-purpose", "cheap_researcher"}
    assert subagents["cheap_researcher"]["model"] is models["deepseek:deepseek-v4-flash"]
    assert subagents["general-purpose"]["model"] is models["openai:gpt-5.6-sol"]
    assert subagents["cheap_researcher"]["tools"] == []
    assert subagents["cheap_researcher"]["permissions"] == parent["permissions"]
    assert subagents["cheap_researcher"]["interrupt_on"] == parent["interrupt_on"]
    assert subagents["cheap_researcher"]["middleware"] is not parent["middleware"]
    assert subagents["cheap_researcher"]["middleware"][0].__class__ is parent["middleware"][0].__class__


def test_nested_delegation_auto_helper_is_explicitly_governed(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_agent_factory(**kwargs: Any) -> FakeGraph:
        captured.update(kwargs)
        return FakeGraph()

    settings = CaroSettings(
        workspace={"root": tmp_path},
        memory={"enabled": False},
        orchestration={
            "enabled": True,
            "allow_nested_delegation": True,
            "agents": {
                "reviewer": {
                    "description": "Review the change.",
                    "system_prompt": "Return a review.",
                    "memory": False,
                }
            },
            "stages": [{"name": "review", "agents": ["reviewer"]}],
        },
    )
    assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=lambda _: FakeListChatModel(responses=["ok"]),
        agent_factory=fake_agent_factory,
    )

    assert [spec["name"] for spec in captured["subagents"]] == ["general-purpose"]
    assert captured["subagents"][0]["interrupt_on"] == captured["interrupt_on"]
    assert captured["subagents"][0]["middleware"][0].__class__ is captured["middleware"][0].__class__


def test_role_task_invokes_the_configured_delegate_model(tmp_path: Path) -> None:
    models: dict[str, BindableFakeMessagesModel] = {
        "openai:gpt-5.6-sol": BindableFakeMessagesModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Find the parser entry point.",
                                "subagent_type": "cheap_researcher",
                            },
                            "id": "delegate-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Delegation complete."),
            ]
        ),
        "deepseek:deepseek-v4-flash": BindableFakeMessagesModel(
            responses=[
                AIMessage(content="The parser starts in src/parser.py."),
                AIMessage(content="The parser starts in src/parser.py."),
            ]
        ),
    }

    def fake_model_factory(definition: Any) -> BindableFakeMessagesModel:
        return models[definition.model]

    settings = CaroSettings(
        workspace={"root": tmp_path, "shell_enabled": False},
        memory={"enabled": False},
        approvals={"enabled": False},
        routing={
            "default": "high",
            "models": {
                "high": {"model": "openai:gpt-5.6-sol"},
                "cheap": {"model": "deepseek:deepseek-v4-flash"},
            },
        },
        orchestration={
            "enabled": True,
            "allow_nested_delegation": True,
            "agents": {
                "architect": {
                    "description": "Design the change.",
                    "system_prompt": "Analyze the architecture.",
                    "model": "high",
                    "memory": False,
                    "delegates": [
                        {
                            "name": "cheap_researcher",
                            "description": "Gather focused evidence.",
                            "system_prompt": "Return evidence only.",
                            "model": "cheap",
                        }
                    ],
                }
            },
            "stages": [{"name": "analysis", "agents": ["architect"]}],
        },
    )
    assembly = assemble_agent(
        settings,
        InMemorySaver(),
        model_factory=fake_model_factory,
    )

    result = assembly.role_graphs["architect"].invoke(
        {"messages": [HumanMessage(content="Inspect the parser.")]}
    )

    assert result["messages"][-1].text == "Delegation complete."
    assert models["deepseek:deepseek-v4-flash"].i == 1

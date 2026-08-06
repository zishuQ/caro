"""Tests for the boundary between Caro and the maintained terminal runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from deepagents.backends import FilesystemBackend
from deepagents_code.app import DeepAgentsApp
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from caro.agent import AgentAssembly
from caro.code import build_code_agent_session
from caro.config import CaroSettings, OrchestrationSettings
from caro.memory import ProjectMemory
from caro.memory.store import MemoryStore
from caro.models.pool import ModelPool
from caro.runtime import CaroRuntime
from caro.state.checkpoints import CheckpointDatabase
from caro.state.runs import RunStatus, RunStore
from caro.tools.registry import ToolRegistry
from caro.ui import _finalize_textual_run


class BindableFakeModel(FakeMessagesListChatModel):
    """Fake chat model with the tool-binding surface used by agent graphs."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BindableFakeModel:
        """Keep the fake response model usable by tool-aware middleware."""

        del tools, kwargs
        return self


def _runtime(tmp_path: Path) -> CaroRuntime:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
        code={"mcp_enabled": False},
    )
    memory = ProjectMemory(tmp_path, settings.memory)
    memory.prepare()
    structured = MemoryStore(
        tmp_path / ".caro" / settings.memory.structured_database,
        allowed_root=tmp_path,
    )
    checkpoints = CheckpointDatabase(tmp_path / ".caro" / "checkpoints.sqlite3")
    assembly = AgentAssembly(
        graph=object(),
        pool=ModelPool(
            settings.routing,
            factory=lambda _: FakeListChatModel(responses=["ok"]),
        ),
        registry=ToolRegistry.default(),
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        memory=memory,
        structured_memory=structured,
        role_graphs={},
        orchestrated=False,
    )
    return CaroRuntime(
        settings=settings,
        runs=RunStore(tmp_path / ".caro" / "runs.sqlite3"),
        checkpoints=checkpoints,
        assembly=assembly,
    )


async def test_code_adapter_keeps_terminal_capabilities_in_caro(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_cli_agent(**kwargs: Any) -> tuple[object, object]:
        captured.update(kwargs)
        return object(), "code-backend"

    from deepagents_code import agent as code_agent

    monkeypatch.setattr(code_agent, "create_cli_agent", fake_create_cli_agent)
    monkeypatch.setattr(code_agent, "load_async_subagents", lambda: [])
    runtime = _runtime(tmp_path)
    from deepagents_code.config import settings as code_settings

    previous_identity = (code_settings.model_provider, code_settings.model_name)
    session = await build_code_agent_session(runtime, runtime.settings)
    try:
        assert session.backend == "code-backend"
        assert (code_settings.model_provider, code_settings.model_name) == (
            "openai",
            "gpt-5.6-sol",
        )
        assert isinstance(captured["checkpointer"], AsyncSqliteSaver)
        assert isinstance(captured["store"], InMemoryStore)
        assert captured["enable_skills"] is True
        assert captured["enable_ask_user"] is True
        assert captured["sandbox"] is runtime.assembly.backend
        assert captured["approval_interrupt_on"]["write_file"]
        assert captured["approval_interrupt_on"]["read_file"] is False
        assert captured["approval_interrupt_on"]["memory_propose"]["allowed_decisions"] == [
            "approve",
            "reject",
        ]
        tool_names = {getattr(tool, "name", "") for tool in captured["tools"]}
        assert {"memory_search", "memory_propose", "memory_archive"} <= tool_names
        gated_names = {getattr(tool, "name", "") for tool in captured["mcp_tools"]}
        assert {"memory_propose", "memory_archive"} <= gated_names
    finally:
        await session.close()
        runtime.close()
        code_settings.model_provider, code_settings.model_name = previous_identity


async def test_local_tui_model_switch_uses_caro_resolver() -> None:
    resolver = AsyncMock(return_value="openai:gpt-5.6-terra")
    app = DeepAgentsApp(
        agent=object(),
        local_model_resolver=resolver,
        configured_models=(("openai:gpt-5.6-luna", "openai"),),
    )
    app._mount_message = AsyncMock()  # type: ignore[method-assign]
    app._restore_effort_override = AsyncMock()  # type: ignore[method-assign]

    await app._switch_model("terra", persist=False)

    resolver.assert_awaited_once_with("terra", None)
    assert app._model_override == "openai:gpt-5.6-terra"
    app._mount_message.assert_awaited_once()  # type: ignore[union-attr]


async def test_code_graph_uses_session_model_override(tmp_path: Path) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
        approvals={"enabled": False},
        memory={"enabled": False},
        code={"mcp_enabled": False, "skills_enabled": False, "ask_user_enabled": False},
        routing={
            "default": "luna",
            "models": {
                "luna": {"model": "openai:gpt-5.6-luna"},
                "flash": {"model": "deepseek:deepseek-v4-flash"},
            },
        },
    )

    def factory(definition: Any) -> BindableFakeModel:
        return BindableFakeModel(
            responses=[AIMessage(content=definition.model)] * 3,
        )

    runtime = CaroRuntime.open(settings, model_factory=factory)
    session = await build_code_agent_session(runtime, settings)
    try:
        first = await session.graph.ainvoke(
            {"messages": [{"role": "user", "content": "first"}]},
            {"configurable": {"thread_id": "model-switch"}},
        )
        second = await session.graph.ainvoke(
            {"messages": [{"role": "user", "content": "second"}]},
            {"configurable": {"thread_id": "model-switch"}},
            context={"model": "deepseek:deepseek-v4-flash", "model_params": {}},
        )
        assert first["messages"][-1].content == "openai:gpt-5.6-luna"
        assert second["messages"][-1].content == "deepseek:deepseek-v4-flash"
    finally:
        await session.close()
        runtime.close()


async def test_orchestration_uses_caro_graph_without_replacing_it(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.settings.orchestration = OrchestrationSettings(
        enabled=True,
        agents={
            "reviewer": {
                "description": "Review the requested change.",
                "system_prompt": "Return a concise review.",
            }
        },
        stages=[{"name": "review", "agents": ["reviewer"]}],
    )
    session = None
    from deepagents_code.config import settings as code_settings

    previous_identity = (code_settings.model_provider, code_settings.model_name)
    try:
        session = await build_code_agent_session(runtime, runtime.settings)
        assert session.graph is not runtime.assembly.graph
        assert session.graph.name == "caro_orchestration"
        assert (code_settings.model_provider, code_settings.model_name) == (
            "openai",
            "gpt-5.6-sol",
        )
    finally:
        if session is not None:
            await session.close()
        runtime.close()
        code_settings.model_provider, code_settings.model_name = previous_identity


async def test_code_adapter_executes_with_async_checkpoint(tmp_path: Path) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
        approvals={"enabled": False},
        memory={"enabled": False},
        code={"mcp_enabled": False, "skills_enabled": False, "ask_user_enabled": False},
    )
    runtime = CaroRuntime.open(
        settings,
        model_factory=lambda _: BindableFakeModel(
            responses=[AIMessage(content="async checkpoint ok")]
        ),
    )
    session = await build_code_agent_session(runtime, settings)
    try:
        result = await session.graph.ainvoke(
            {"messages": [{"role": "user", "content": "Say hello."}]},
            {"configurable": {"thread_id": "async-check"}},
        )
        assert result["messages"][-1].content == "async checkpoint ok"
    finally:
        await session.close()
        runtime.close()


@pytest.mark.parametrize(
    ("interrupts", "next_nodes", "return_code", "expected"),
    [
        ((SimpleNamespace(id="approval"),), (), 0, RunStatus.WAITING_APPROVAL),
        ((), ("model",), 0, RunStatus.PAUSED),
        ((), (), 0, RunStatus.COMPLETED),
        ((), (), 1, RunStatus.FAILED),
    ],
)
async def test_textual_finalization_preserves_graph_lifecycle_state(
    tmp_path: Path,
    interrupts: tuple[Any, ...],
    next_nodes: tuple[str, ...],
    return_code: int,
    expected: RunStatus,
) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        state={"directory": tmp_path / ".caro"},
    )
    store = RunStore(tmp_path / ".caro" / "runs.sqlite3")
    run = store.create(task="interactive", workspace=tmp_path, config_hash=settings.fingerprint())
    run = store.transition(run.run_id, RunStatus.RUNNING)

    class Graph:
        async def aget_state(self, config: Any) -> Any:
            assert config["configurable"]["thread_id"] == run.thread_id
            return SimpleNamespace(interrupts=interrupts, next=next_nodes)

    session = SimpleNamespace(run=run, graph=Graph())
    result = SimpleNamespace(return_code=return_code, thread_id=run.thread_id)
    await _finalize_textual_run(SimpleNamespace(runs=store), session, result)

    assert store.get(run.run_id).status is expected

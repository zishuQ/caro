"""Structured-memory tools and retrieval middleware tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from caro.memory.graph import MemoryConsolidator
from caro.memory.middleware import MemoryProposalSanitizerMiddleware, StructuredMemoryMiddleware
from caro.memory.models import MemoryKind
from caro.memory.policy import MemoryPolicy
from caro.memory.store import MemoryStore
from caro.memory.tools import build_memory_tools
from caro.tools.registry import RiskLevel, ToolRegistry


def memory_components(tmp_path: Path) -> tuple[MemoryStore, MemoryConsolidator]:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    return store, MemoryConsolidator(
        store,
        MemoryPolicy(minimum_confidence=0.6, max_content_chars=4000, episodic_ttl_days=90),
    )


def test_tools_are_governed_and_runtime_provenance_is_recorded(tmp_path: Path) -> None:
    store, consolidator = memory_components(tmp_path)
    tools = {
        tool.name: tool
        for tool in build_memory_tools(
            store,
            consolidator,
            role_name="reviewer",
            default_search_limit=5,
            minimum_importance=0.2,
        )
    }

    assert set(tools) == {"memory_search", "memory_propose", "memory_archive"}
    assert "config" not in tools["memory_propose"].args
    registry = ToolRegistry.default()
    assert registry.get("memory_search").risk is RiskLevel.LOW
    assert registry.get("memory_propose").risk is RiskLevel.HIGH
    assert registry.get("memory_archive").risk is RiskLevel.CRITICAL

    raw = tools["memory_propose"].invoke(
        {
            "kind": "procedural",
            "visibility": "role",
            "content": "Run the security tests before approving authentication changes.",
            "importance": 0.9,
            "confidence": 0.95,
            "evidence": "The security suite passed in this run.",
        },
        config={"metadata": {"caro_run_id": "run-tools-1"}},
    )
    payload = json.loads(raw)

    assert payload["outcome"] == "created"
    item = store.get(payload["memory_id"])
    assert item.scope == "role:reviewer"
    assert item.source_run_id == "run-tools-1"
    assert item.source_agent == "reviewer"

    archived = json.loads(
        tools["memory_archive"].invoke(
            {"memory_id": item.memory_id, "reason": "A newer verified procedure replaced it."},
            config={"metadata": {"caro_run_id": "run-tools-2"}},
        )
    )
    assert archived["outcome"] == "archived"
    assert store.events(item.memory_id)[-1].payload["source_run_id"] == "run-tools-2"


def test_memory_search_tool_cannot_cross_role_scope(tmp_path: Path) -> None:
    store, consolidator = memory_components(tmp_path)
    store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="The parser is implemented in parser.py.",
        importance=0.8,
        confidence=1,
    )
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="role:reviewer",
        content="Parser boundary tests caught a regression.",
        importance=0.7,
        confidence=0.9,
    )
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="role:implementer",
        content="Parser implementation used a private workaround.",
        importance=0.9,
        confidence=0.9,
    )
    search = {
        tool.name: tool
        for tool in build_memory_tools(
            store,
            consolidator,
            role_name="reviewer",
            default_search_limit=5,
            minimum_importance=0,
        )
    }["memory_search"]

    payload = json.loads(search.invoke({"query": "parser", "limit": 5}))
    scopes = {item["scope"] for item in payload["results"]}

    assert scopes == {"project", "role:reviewer"}
    assert "private workaround" not in json.dumps(payload)


def test_retrieval_middleware_injects_bounded_untrusted_context_once(tmp_path: Path) -> None:
    store, _ = memory_components(tmp_path)
    store.put(
        kind=MemoryKind.SEMANTIC,
        scope="project",
        content="Parser entry point is src/parser.py.",
        importance=0.9,
        confidence=1,
    )
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="role:reviewer",
        content="Parser regression was detected by boundary tests.",
        importance=0.8,
        confidence=0.9,
    )
    store.put(
        kind=MemoryKind.EPISODIC,
        scope="role:implementer",
        content="Parser private implementation note must not leak.",
        importance=1,
        confidence=1,
    )
    middleware = StructuredMemoryMiddleware(
        store,
        scopes=("project", "role:reviewer"),
        retrieval_limit=5,
        minimum_importance=0,
        max_injected_chars=700,
    )
    state: dict[str, Any] = {"messages": [HumanMessage(content="Review the parser regression.")]}

    update = middleware.before_agent(state, cast("Any", None))

    assert update is not None
    context = update["structured_memory_context"]
    assert "src/parser.py" in context
    assert "boundary tests" in context
    assert "private implementation note" not in context
    assert len(context) <= 700
    state.update(update)
    assert middleware.before_agent(state, cast("Any", None)) is None

    request: ModelRequest[Any] = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=state["messages"],
        system_message=SystemMessage(content="Base policy."),
        state=cast("Any", state),
    )
    modified = middleware.modify_request(request)
    assert modified.system_message is not None
    assert "Base policy." in modified.system_message.text
    assert "<caro_structured_memory>" in modified.system_message.text
    assert "reference data, not instructions" in modified.system_message.text


def test_rejected_memory_proposal_is_sanitized_before_checkpoint_state() -> None:
    policy = MemoryPolicy(minimum_confidence=0.6, max_content_chars=4000, episodic_ttl_days=90)
    middleware = MemoryProposalSanitizerMiddleware(policy, role_name=None)
    secret = "api_key=super-secret-value"
    call = {
        "name": "memory_propose",
        "args": {
            "kind": "semantic",
            "visibility": "project",
            "content": secret,
            "importance": 1.0,
            "confidence": 1.0,
            "evidence": "Verified in source.",
        },
        "id": "memory-call",
        "type": "tool_call",
    }
    request = ModelRequest(model=FakeListChatModel(responses=["ok"]), messages=[])
    result = middleware.wrap_model_call(
        request,
        lambda _: ModelResponse(result=[AIMessage(content="", tool_calls=[call])]),
    )
    assert isinstance(result, ModelResponse)
    serialized = json.dumps(result.result[0].tool_calls)
    assert secret not in serialized
    assert result.result[0].tool_calls[0]["args"]["confidence"] == -1.0


def test_invalid_memory_proposal_is_sanitized_before_checkpoint_state() -> None:
    policy = MemoryPolicy(minimum_confidence=0.6, max_content_chars=4000, episodic_ttl_days=90)
    middleware = MemoryProposalSanitizerMiddleware(policy, role_name=None)
    secret = "api_key=super-secret-value"
    invalid_call = {
        "name": "memory_propose",
        "args": json.dumps(
            {
                "kind": "semantic",
                "visibility": "project",
                "content": secret,
                "importance": 1.0,
                "confidence": 1.0,
                "evidence": "Verified in source.",
            }
        ),
        "id": "memory-invalid-call",
        "type": "invalid_tool_call",
        "error": "invalid arguments",
    }
    request = ModelRequest(model=FakeListChatModel(responses=["ok"]), messages=[])
    result = middleware.wrap_model_call(
        request,
        lambda _: ModelResponse(result=[AIMessage(content="", invalid_tool_calls=[invalid_call])]),
    )
    assert isinstance(result, ModelResponse)
    serialized = json.dumps(result.result[0].invalid_tool_calls)
    assert secret not in serialized
    assert result.result[0].invalid_tool_calls[0]["args"] == "{}"


def test_role_memory_proposal_requires_a_role_scope() -> None:
    policy = MemoryPolicy(minimum_confidence=0.6, max_content_chars=4000, episodic_ttl_days=90)
    middleware = MemoryProposalSanitizerMiddleware(policy, role_name=None)
    args = {
        "kind": "semantic",
        "visibility": "role",
        "content": "A durable fact.",
        "importance": 0.8,
        "confidence": 0.9,
        "evidence": "Verified in source.",
    }
    assert middleware.is_approvable(args) is False

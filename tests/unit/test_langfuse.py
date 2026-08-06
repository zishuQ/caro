"""Langfuse v4 adapter tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from caro.config import CaroSettings
from caro.errors import CaroError
from caro.observability import langfuse as langfuse_adapter
from caro.observability.langfuse import build_trace_context
from caro.state.runs import RunStore


def make_run(tmp_path: Path):
    return RunStore(tmp_path / "runs.sqlite3").create(task="task", workspace=tmp_path, config_hash="hash")


def test_disabled_langfuse_is_noop(tmp_path: Path) -> None:
    settings = CaroSettings(workspace={"root": tmp_path})
    context = build_trace_context(settings, make_run(tmp_path), environ={})
    assert context.callbacks == []
    assert context.metadata == {}


def test_enabled_langfuse_requires_environment_credentials(tmp_path: Path) -> None:
    settings = CaroSettings(workspace={"root": tmp_path}, langfuse={"enabled": True})
    with pytest.raises(CaroError, match="LANGFUSE_PUBLIC_KEY"):
        build_trace_context(settings, make_run(tmp_path), environ={})


def test_v4_callback_metadata_and_flush(tmp_path: Path) -> None:
    settings = CaroSettings(
        workspace={"root": tmp_path},
        langfuse={"enabled": True, "environment": "test", "tags": ["caro-test"]},
    )
    flushed: list[bool] = []
    handler = BaseCallbackHandler()
    context = build_trace_context(
        settings,
        make_run(tmp_path),
        environ={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
        handler_factory=lambda: handler,
        flush=lambda: flushed.append(True),
    )
    assert context.callbacks == [handler]
    assert context.metadata["langfuse_tags"] == [
        "caro-test",
        "deepagents",
        "langgraph",
        "structured-memory",
    ]
    assert context.metadata["caro_memory_enabled"] is True
    assert context.metadata["caro_memory_auto_retrieve"] is True
    context.close()
    assert flushed == [True]


def test_default_structured_client_uses_caro_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        SimpleNamespace(Langfuse=FakeClient),
    )
    langfuse_adapter._default_observation_sink()
    assert captured["mask"] is langfuse_adapter._langfuse_mask


def test_default_handler_binds_public_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCallbackHandler:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langfuse.langchain",
        SimpleNamespace(CallbackHandler=FakeCallbackHandler),
    )
    langfuse_adapter._default_handler(public_key="pk")
    assert captured == {"public_key": "pk"}

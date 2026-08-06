"""Model initialization diagnostics tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from caro.config import RoutingSettings
from caro.errors import ProviderInitializationError
from caro.models import pool as pool_module
from caro.models.pool import ModelPool


def test_default_factory_forwards_full_identifier_and_typed_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> FakeListChatModel:
        captured.update({"model": model, **kwargs})
        return FakeListChatModel(responses=["ok"])

    monkeypatch.setattr(pool_module, "init_chat_model", fake_init)
    settings = RoutingSettings.model_validate(
        {
            "default": "ds",
            "models": {
                "ds": {
                    "model": "deepseek:deepseek-v4-flash",
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "max_retries": 4,
                    "options": {"top_p": 0.8},
                }
            },
        }
    )

    ModelPool(settings).get("ds")

    assert captured == {
        "model": "deepseek:deepseek-v4-flash",
        "temperature": 0.2,
        "max_tokens": 4096,
        "max_retries": 4,
        "top_p": 0.8,
    }


def test_default_factory_applies_deepagents_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI models inherit DeepAgents' Responses API default."""
    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> FakeListChatModel:
        captured.update({"model": model, **kwargs})
        return FakeListChatModel(responses=["ok"])

    monkeypatch.setattr(pool_module, "init_chat_model", fake_init)
    settings = RoutingSettings.model_validate(
        {
            "default": "openai",
            "models": {"openai": {"model": "openai:gpt-5.6-luna"}},
        }
    )

    ModelPool(settings).get("openai")

    assert captured["use_responses_api"] is True


def test_default_factory_honors_explicit_responses_api_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A YAML option can explicitly opt out of the provider default."""
    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> FakeListChatModel:
        captured.update({"model": model, **kwargs})
        return FakeListChatModel(responses=["ok"])

    monkeypatch.setattr(pool_module, "init_chat_model", fake_init)
    settings = RoutingSettings.model_validate(
        {
            "default": "openai",
            "models": {
                "openai": {
                    "model": "openai:gpt-5.6-luna",
                    "options": {"use_responses_api": False},
                }
            },
        }
    )

    ModelPool(settings).get("openai")

    assert captured["use_responses_api"] is False


def test_default_factory_uses_chat_completions_for_openai_compatible_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom OpenAI endpoints default to the protocol relays commonly expose."""
    captured: dict[str, Any] = {}

    def fake_init(model: str, **kwargs: Any) -> FakeListChatModel:
        captured.update({"model": model, **kwargs})
        return FakeListChatModel(responses=["ok"])

    monkeypatch.setattr(pool_module, "init_chat_model", fake_init)
    settings = RoutingSettings.model_validate(
        {
            "default": "relay",
            "models": {
                "relay": {
                    "model": "openai:gpt-5.6-luna",
                    "options": {"base_url": "https://relay.example/v1"},
                }
            },
        }
    )

    ModelPool(settings).get("relay")

    assert captured["use_responses_api"] is False


def test_initialization_error_contains_provider_specific_remediation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: str, **__: Any) -> FakeListChatModel:
        raise ImportError("missing integration")

    monkeypatch.setattr(pool_module, "init_chat_model", fail)
    settings = RoutingSettings.model_validate(
        {"default": "ds", "models": {"ds": {"model": "deepseek:deepseek-v4-flash"}}}
    )

    with pytest.raises(
        ProviderInitializationError,
        match=r"langchain-deepseek is installed|uv sync --extra deepseek",
    ):
        ModelPool(settings).get("ds")


def test_pool_resolves_names_and_canonical_identifiers() -> None:
    settings = RoutingSettings.model_validate(
        {
            "default": "luna",
            "models": {
                "luna": {"model": "openai:gpt-5.6-luna"},
                "flash": {"model": "deepseek:deepseek-v4-flash"},
            },
        }
    )
    pool = ModelPool(settings, factory=lambda _: FakeListChatModel(responses=["ok"]))

    assert pool.resolve_name("luna") == "luna"
    assert pool.resolve_name("openai:gpt-5.6-luna") == "luna"
    assert pool.canonical_identifier("flash") == "deepseek:deepseek-v4-flash"
    with pytest.raises(KeyError, match="choose one of: luna, flash"):
        pool.resolve_name("openai:gpt-5.6-terra")

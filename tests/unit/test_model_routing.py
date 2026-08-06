"""Rule routing and fallback adapter tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from caro.config import RoutingSettings
from caro.errors import ProviderInitializationError
from caro.models.pool import ModelPool
from caro.models.routing import RuleRouter, RuleRoutingMiddleware, build_model_middleware


def routing_settings() -> RoutingSettings:
    return RoutingSettings.model_validate(
        {
            "default": "openai",
            "models": {
                "openai": {"model": "openai:gpt-5.6-sol"},
                "deepseek": {"model": "deepseek:deepseek-v4-flash"},
                "backup": {"model": "anthropic:claude-sonnet-4-6"},
            },
            "rules": [
                {
                    "name": "docs-on-deepseek",
                    "model": "deepseek",
                    "match": {"any_keywords": ["README", "文档"], "max_prompt_chars": 200},
                }
            ],
            "fallbacks": ["backup"],
        }
    )


class BindableFakeModel(FakeListChatModel):
    """Fake model that advertises the tool-binding capability Caro requires."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BindableFakeModel:
        """Return the fake model after accepting a local capability probe."""

        del tools, kwargs
        return self


def fake_factory(definition: Any) -> BaseChatModel:
    return BindableFakeModel(responses=[definition.model])


def make_request(model: BaseChatModel, prompt: str) -> ModelRequest[Any]:
    return ModelRequest(model=model, messages=[HumanMessage(prompt)])


def test_explicit_session_model_override_precedes_prompt_rules() -> None:
    pool = ModelPool(routing_settings(), factory=fake_factory)
    router = RuleRoutingMiddleware(pool, RuleRouter(pool.settings))
    request = ModelRequest(
        model=pool.get("openai"),
        messages=[HumanMessage("README 文档")],
        runtime=Runtime(context={"model": "openai:gpt-5.6-sol"}),
    )

    result = router.wrap_model_call(
        request,
        lambda routed: ModelResponse(
            result=[AIMessage(content=str(routed.model is pool.get("openai")))]
        ),
    )

    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "True"


def test_first_matching_rule_routes_by_name() -> None:
    router = RuleRouter(routing_settings())
    assert router.decide("请更新 README 文档").model_name == "deepseek"
    assert router.decide("implement the parser").model_name == "openai"


def test_route_updates_model_identity_prompt() -> None:
    """Rule-routed requests identify the model that will actually answer."""
    pool = ModelPool(routing_settings(), factory=fake_factory)
    router = RuleRoutingMiddleware(pool, RuleRouter(pool.settings))
    request = ModelRequest(
        model=pool.get("openai"),
        messages=[HumanMessage("README")],
        system_prompt=(
            "### Model Identity\n\n"
            "You are running as model `gpt-5.6-sol` (provider: openai).\n\n"
            "### Tools\n\nUse tools.\n"
        ),
    )
    captured: list[ModelRequest[Any]] = []

    result = router.wrap_model_call(
        request,
        lambda routed: (captured.append(routed), ModelResponse(result=[AIMessage("ok")]))[1],
    )

    assert isinstance(result, ModelResponse)
    assert captured[0].model is pool.get("deepseek")
    assert "`deepseek-v4-flash`" in (captured[0].system_prompt or "")
    assert "`gpt-5.6-sol`" not in (captured[0].system_prompt or "")


def test_rule_must_satisfy_every_condition() -> None:
    router = RuleRouter(routing_settings())
    assert router.decide("README " + ("x" * 300)).model_name == "openai"


def test_invalid_regex_fails_at_router_construction() -> None:
    settings = RoutingSettings.model_validate(
        {
            "models": {"primary": {"model": "openai:gpt-5.6-sol"}},
            "rules": [{"name": "bad", "model": "primary", "match": {"regex": "["}}],
        }
    )
    with pytest.raises(Exception, match="unterminated"):
        RuleRouter(settings)


def test_model_pool_is_lazy_and_caches_instances() -> None:
    calls: list[str] = []

    def factory(definition: Any) -> BaseChatModel:
        calls.append(definition.model)
        return fake_factory(definition)

    pool = ModelPool(routing_settings(), factory=factory)
    assert calls == []
    assert pool.get("openai") is pool.get("openai")
    assert calls == ["openai:gpt-5.6-sol"]


def test_routing_wraps_maintained_fallback_for_cross_provider_failure() -> None:
    pool = ModelPool(routing_settings(), factory=fake_factory)
    middleware = build_model_middleware(pool)
    router = middleware[0]
    fallback = middleware[1]
    assert isinstance(router, RuleRoutingMiddleware)
    assert isinstance(fallback, ModelFallbackMiddleware)

    attempted: list[BaseChatModel] = []

    def final_handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        attempted.append(request.model)
        if len(attempted) == 1:
            raise RuntimeError("provider outage")
        return ModelResponse(result=[AIMessage("fallback worked")])

    def fallback_handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        return fallback.wrap_model_call(request, final_handler)  # type: ignore[return-value]

    result = router.wrap_model_call(make_request(pool.get("openai"), "README"), fallback_handler)

    assert isinstance(result, ModelResponse)
    assert result.result[0].text == "fallback worked"
    assert attempted == [pool.get("deepseek"), pool.get("backup")]


def test_decision_observer_receives_rule_without_prompt_content() -> None:
    decisions: list[str | None] = []
    pool = ModelPool(routing_settings(), factory=fake_factory)
    middleware = RuleRoutingMiddleware(
        pool,
        RuleRouter(pool.settings),
        on_decision=lambda decision: decisions.append(decision.rule_name),
    )
    def handler(_: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(result=[AIMessage("ok")])

    middleware.wrap_model_call(make_request(pool.get("openai"), "README secret"), handler)
    assert decisions == ["docs-on-deepseek"]


def test_unavailable_fallback_does_not_break_middleware_construction() -> None:
    def factory(definition: Any) -> BaseChatModel:
        if definition.model.startswith("anthropic:"):
            raise ProviderInitializationError("backup credentials unavailable")
        return fake_factory(definition)

    pool = ModelPool(routing_settings(), factory=factory)
    middleware = build_model_middleware(pool)
    assert len(middleware) == 1


def test_fallback_without_tool_binding_is_skipped() -> None:
    def factory(definition: Any) -> BaseChatModel:
        if definition.model.startswith("anthropic:"):
            return FakeListChatModel(responses=[definition.model])
        return fake_factory(definition)

    pool = ModelPool(routing_settings(), factory=factory)
    middleware = build_model_middleware(pool)
    assert len(middleware) == 1


def test_rule_route_without_tool_binding_uses_compatible_fallback() -> None:
    def factory(definition: Any) -> BaseChatModel:
        if definition.model.startswith("deepseek:"):
            return FakeListChatModel(responses=[definition.model])
        return fake_factory(definition)

    pool = ModelPool(routing_settings(), factory=factory)
    router = RuleRoutingMiddleware(pool, RuleRouter(pool.settings))

    result = router.wrap_model_call(
        make_request(pool.get("openai"), "README"),
        lambda request: ModelResponse(result=[AIMessage(request.model.__class__.__name__)]),
    )

    assert isinstance(result, ModelResponse)
    assert result.result[0].text == "BindableFakeModel"


def test_rule_model_initialization_failure_uses_configured_fallback() -> None:
    def factory(definition: Any) -> BaseChatModel:
        if definition.model.startswith("deepseek:"):
            raise ProviderInitializationError("route credentials unavailable")
        return fake_factory(definition)

    pool = ModelPool(routing_settings(), factory=factory)
    router = RuleRoutingMiddleware(pool, RuleRouter(pool.settings))

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(result=[AIMessage(request.model.__class__.__name__)])

    result = router.wrap_model_call(make_request(pool.get("openai"), "README"), handler)
    assert isinstance(result, ModelResponse)
    assert pool.get("backup") is not pool.get("openai")

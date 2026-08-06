"""Deterministic model routing middleware with LangChain fallback adaptation."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from deepagents_code.configurable_model import patch_model_identity
from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from caro.config import RoutingRule, RoutingSettings
from caro.errors import ProviderInitializationError
from caro.models.pool import ModelPool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Auditable result of applying ordered routing rules."""

    model_name: str
    rule_name: str | None


def _rule_matches(rule: RoutingRule, prompt: str) -> bool:
    """Return whether all configured rule conditions match a prompt."""

    condition = rule.match
    prompt_length = len(prompt)
    if condition.any_keywords and not any(
        keyword.casefold() in prompt.casefold() for keyword in condition.any_keywords
    ):
        return False
    if condition.regex is not None and re.search(condition.regex, prompt, flags=re.IGNORECASE) is None:
        return False
    if condition.min_prompt_chars is not None and prompt_length < condition.min_prompt_chars:
        return False
    return condition.max_prompt_chars is None or prompt_length <= condition.max_prompt_chars


class RuleRouter:
    """Apply the first matching routing rule, otherwise select the default."""

    def __init__(self, settings: RoutingSettings) -> None:
        self._settings = settings
        for rule in settings.rules:
            if rule.match.regex is not None:
                re.compile(rule.match.regex)

    def decide(self, prompt: str) -> RoutingDecision:
        """Select a named model for a prompt."""

        for rule in self._settings.rules:
            if _rule_matches(rule, prompt):
                return RoutingDecision(model_name=rule.model, rule_name=rule.name)
        return RoutingDecision(model_name=self._settings.default, rule_name=None)


def request_prompt(request: ModelRequest[Any]) -> str:
    """Extract the most recent human prompt without serializing tool output."""

    for message in reversed(request.messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


DecisionObserver = Callable[[RoutingDecision], None]

_TOOL_BINDING_PROBE = {
    "name": "caro_tool_binding_probe",
    "description": "Internal capability probe; never sent to a provider request.",
    "parameters": {"type": "object", "properties": {}},
}


def _supports_tool_binding(model: Any) -> bool:
    """Check tool binding locally before a model enters an agent fallback path."""

    bind_tools = getattr(model, "bind_tools", None)
    if not callable(bind_tools):
        return False
    try:
        return bind_tools([_TOOL_BINDING_PROBE]) is not None
    except Exception as error:  # Provider capability errors must fail closed.
        logger.warning(
            "Skipping model without usable tool binding (%s)",
            type(error).__name__,
        )
        return False


class RuleRoutingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Override each model request with the model selected by Caro rules."""

    def __init__(
        self,
        pool: ModelPool,
        router: RuleRouter,
        *,
        on_decision: DecisionObserver | None = None,
    ) -> None:
        super().__init__()
        self._pool = pool
        self._router = router
        self._on_decision = on_decision

    def _route_model(self, request: ModelRequest[Any], model_name: str) -> ModelRequest[Any]:
        """Apply a pool model and keep the request identity prompt in sync."""

        model = self._pool.get(model_name)
        model_spec = self._pool.definition(model_name).model
        routed = request.override(model=model)
        return patch_model_identity(routed, model_spec)

    def _route(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        decision = self._router.decide(request_prompt(request))
        logger.info("Caro model route: model=%s rule=%s", decision.model_name, decision.rule_name or "default")
        if self._on_decision is not None:
            self._on_decision(decision)
        return self._route_model(request, decision.model_name)

    @staticmethod
    def _explicit_model(request: ModelRequest[Any]) -> str | None:
        """Read a per-invocation model override from Caro/Code context."""

        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        value: object = (
            context.get("model")
            if isinstance(context, dict)
            else getattr(context, "model", None)
        )
        return value if isinstance(value, str) and value else None

    def _route_with_explicit_override(
        self, request: ModelRequest[Any]
    ) -> ModelRequest[Any] | None:
        """Resolve a user-selected Caro model before applying prompt rules."""

        override = self._explicit_model(request)
        if override is None:
            return None
        try:
            model_name = self._pool.resolve_name(override)
        except (KeyError, ValueError) as error:
            raise ProviderInitializationError(str(error)) from error
        decision = RoutingDecision(model_name=model_name, rule_name="session-override")
        logger.info("Caro model override: model=%s", model_name)
        if self._on_decision is not None:
            self._on_decision(decision)
        return self._route_model(request, decision.model_name)

    def _route_with_initialization_fallback(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Route a request while tolerating an unavailable rule-selected model."""

        explicit = self._route_with_explicit_override(request)
        if explicit is not None:
            return explicit

        decision = self._router.decide(request_prompt(request))
        logger.info("Caro model route: model=%s rule=%s", decision.model_name, decision.rule_name or "default")
        if self._on_decision is not None:
            self._on_decision(decision)
        try:
            model = self._pool.get(decision.model_name)
            if not _supports_tool_binding(model):
                raise ProviderInitializationError(
                    f"model {decision.model_name!r} does not support tool binding required by Caro"
                )
        except Exception as error:
            logger.warning(
                "Caro model route %s could not initialize; trying configured fallbacks (%s)",
                decision.model_name,
                type(error).__name__,
            )
            for fallback_name in self._pool.settings.fallbacks:
                if fallback_name == decision.model_name:
                    continue
                try:
                    fallback_model = self._pool.get(fallback_name)
                    if not _supports_tool_binding(fallback_model):
                        raise ProviderInitializationError(
                            f"model {fallback_name!r} does not support tool binding required by Caro"
                        )
                    return self._route_model(request, fallback_name)
                except Exception as fallback_error:
                    logger.warning(
                        "Caro fallback model %s could not initialize (%s)",
                        fallback_name,
                        type(fallback_error).__name__,
                    )
            raise
        return self._route_model(request, decision.model_name)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        """Route a synchronous model request."""

        return handler(self._route_with_initialization_fallback(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        """Route an asynchronous model request."""

        return await handler(self._route_with_initialization_fallback(request))


def build_model_middleware(
    pool: ModelPool,
    *,
    on_decision: DecisionObserver | None = None,
) -> Sequence[AgentMiddleware[Any, Any, Any]]:
    """Build Caro routing plus LangChain's maintained failure fallback middleware."""

    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        RuleRoutingMiddleware(pool, RuleRouter(pool.settings), on_decision=on_decision)
    ]
    middleware.extend(build_fallback_middleware(pool))
    return middleware


def build_fallback_middleware(
    pool: ModelPool,
    *,
    exclude_names: frozenset[str] = frozenset(),
) -> Sequence[AgentMiddleware[Any, Any, Any]]:
    """Build only maintained failure fallback behavior for fixed-role agents."""

    fallback_models = []
    for name in pool.settings.fallbacks:
        if name in exclude_names:
            continue
        try:
            fallback_model = pool.get(name)
            if not _supports_tool_binding(fallback_model):
                logger.warning(
                    "Skipping fallback model %s because it does not support Caro tool binding",
                    name,
                )
                continue
            fallback_models.append(fallback_model)
        except Exception as error:
            # Fallback providers are optional. An unavailable backup must not
            # prevent the primary graph from being assembled.
            logger.warning(
                "Skipping unavailable Caro fallback model %s (%s)",
                name,
                type(error).__name__,
            )
    if not fallback_models:
        return ()
    return (ModelFallbackMiddleware(fallback_models[0], *fallback_models[1:]),)

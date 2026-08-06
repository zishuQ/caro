"""Lazy, named chat-model pool."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from deepagents.profiles.provider.provider_profiles import apply_provider_profile
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from caro.config import ModelDefinition, RoutingSettings
from caro.errors import ProviderInitializationError
from caro.models.providers import (
    get_provider_spec,
    parse_model_identifier,
    provider_status,
)

ModelFactory = Callable[[ModelDefinition], BaseChatModel]


def _default_factory(definition: ModelDefinition) -> BaseChatModel:
    """Construct a model through LangChain's current public factory."""

    try:
        kwargs = definition.init_kwargs()
        identifier = parse_model_identifier(definition.model)
        # Deep Agents enables OpenAI's Responses API by default. Most
        # OpenAI-compatible relays only implement Chat Completions, so choose
        # that protocol automatically when a custom endpoint is configured;
        # an explicit `use_responses_api` option still takes precedence.
        if (
            identifier.provider == "openai"
            and any(key in kwargs for key in ("base_url", "openai_api_base"))
            and "use_responses_api" not in kwargs
        ):
            kwargs["use_responses_api"] = False
        kwargs = apply_provider_profile(definition.model, kwargs)
        return cast("BaseChatModel", init_chat_model(definition.model, **kwargs))
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        identifier = parse_model_identifier(definition.model)
        spec = get_provider_spec(identifier.provider)
        if spec is None:
            detail = "install the LangChain integration package for this provider and configure its credentials"
        else:
            credentials = "/".join(spec.credential_variables) if spec.credential_variables else "no API key"
            status = provider_status(identifier.provider)
            if status.installed:
                detail = (
                    f"integration {spec.distribution} is installed; "
                    f"check credentials/client configuration ({credentials})"
                )
            else:
                detail = f"run `{spec.install_hint}`; credential source: {credentials}"
        raise ProviderInitializationError(
            f"cannot initialize model {definition.model!r}: {detail}. Original error: {error}"
        ) from error


class ModelPool:
    """Lazily initialize configured models once and address them by stable name."""

    def __init__(self, settings: RoutingSettings, *, factory: ModelFactory | None = None) -> None:
        self._settings = settings
        self._factory = _default_factory if factory is None else factory
        self._instances: dict[str, BaseChatModel] = {}

    @property
    def settings(self) -> RoutingSettings:
        """Return immutable-by-convention routing configuration."""

        return self._settings

    def definition(self, name: str) -> ModelDefinition:
        """Return one named model definition."""

        try:
            return self._settings.models[name]
        except KeyError as error:
            raise KeyError(f"unknown model pool entry: {name}") from error

    def resolve_name(self, value: str) -> str:
        """Resolve a configured pool name or model identifier to a pool name.

        Interactive model selection is intentionally limited to models declared
        in Caro's routing pool. Accepting both the friendly pool name and the
        canonical ``provider:model`` identifier keeps `/model primary` and
        `/model openai:gpt-5.6-luna` equivalent without allowing the TUI to
        bypass Caro's provider and endpoint configuration.

        Args:
            value: Configured pool name or canonical model identifier.

        Returns:
            The matching configured pool name.

        Raises:
            KeyError: If the value is not present in the configured pool.
            ValueError: If a model identifier is malformed.
        """
        if value in self._settings.models:
            return value
        identifier = parse_model_identifier(value)
        for name, definition in self._settings.models.items():
            if definition.model == identifier.value:
                return name
        configured = ", ".join(self._settings.models)
        raise KeyError(
            f"model {value!r} is not configured in Caro's model pool; "
            f"choose one of: {configured}"
        )

    def canonical_identifier(self, value: str) -> str:
        """Return the canonical identifier for a configured model selection."""

        return self.definition(self.resolve_name(value)).model

    def get_for(self, value: str) -> BaseChatModel:
        """Return the model instance selected by pool name or identifier."""

        return self.get(self.resolve_name(value))

    def get(self, name: str) -> BaseChatModel:
        """Return a cached model instance, constructing it on first access."""

        if name not in self._instances:
            self._instances[name] = self._factory(self.definition(name))
        return self._instances[name]

    def configured_names(self) -> tuple[str, ...]:
        """Return configured pool names in YAML order."""

        return tuple(self._settings.models)

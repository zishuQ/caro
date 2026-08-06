"""Provider catalog and model-identifier validation.

Caro delegates model construction to LangChain's public ``init_chat_model``
interface.  This module adds a stable governance view: known integration
packages, credential diagnostics, aliases, and agent-specific capability
checks. Unknown LangChain providers remain usable and are reported as
unmanaged rather than rejected.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import Literal

CredentialPolicy = Literal["required", "optional", "none"]
CredentialState = Literal["configured", "missing", "external", "local", "unmanaged"]

_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "gemini": "google_genai",
        "google": "google_genai",
    }
)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Metadata for one explicitly supported LangChain provider integration."""

    provider: str
    display_name: str
    distribution: str
    import_name: str
    extra: str
    credential_variables: tuple[str, ...]
    credential_policy: CredentialPolicy = "required"
    note: str = ""

    @property
    def install_hint(self) -> str:
        """Return the project installation command for this provider."""

        return f"uv sync --extra {self.extra}"


@dataclass(frozen=True, slots=True)
class ModelIdentifier:
    """A parsed ``provider:model`` identifier."""

    provider: str
    model: str

    @property
    def value(self) -> str:
        """Return the canonical model identifier."""

        return f"{self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Local readiness of a provider used by configured models."""

    provider: str
    spec: ProviderSpec | None
    installed_version: str | None
    credential_state: CredentialState
    credential_variables: tuple[str, ...]

    @property
    def installed(self) -> bool | None:
        """Return install state, or ``None`` for an unmanaged provider."""

        if self.spec is None:
            return None
        return self.installed_version is not None

    @property
    def ready(self) -> bool | None:
        """Return a conservative readiness result.

        ``None`` means Caro cannot prove readiness, for example when Google ADC
        or an unknown third-party integration may provide credentials.
        """

        if self.spec is None:
            return None
        if self.installed_version is None:
            return False
        if self.credential_state == "missing":
            return False
        if self.credential_state in {"external", "unmanaged"}:
            return None
        return True


_PROVIDER_ITEMS = (
    ProviderSpec(
        provider="openai",
        display_name="OpenAI",
        distribution="langchain-openai",
        import_name="langchain_openai",
        extra="openai",
        credential_variables=("OPENAI_API_KEY",),
    ),
    ProviderSpec(
        provider="deepseek",
        display_name="DeepSeek",
        distribution="langchain-deepseek",
        import_name="langchain_deepseek",
        extra="deepseek",
        credential_variables=("DEEPSEEK_API_KEY",),
        note="Use deepseek-v4-flash for tool-calling agents; deepseek-reasoner is unsupported.",
    ),
    ProviderSpec(
        provider="anthropic",
        display_name="Anthropic",
        distribution="langchain-anthropic",
        import_name="langchain_anthropic",
        extra="anthropic",
        credential_variables=("ANTHROPIC_API_KEY",),
    ),
    ProviderSpec(
        provider="google_genai",
        display_name="Google Gemini",
        distribution="langchain-google-genai",
        import_name="langchain_google_genai",
        extra="google",
        credential_variables=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        credential_policy="optional",
        note="API keys are detected; Google Application Default Credentials may also be used.",
    ),
    ProviderSpec(
        provider="openrouter",
        display_name="OpenRouter",
        distribution="langchain-openrouter",
        import_name="langchain_openrouter",
        extra="openrouter",
        credential_variables=("OPENROUTER_API_KEY",),
    ),
    ProviderSpec(
        provider="ollama",
        display_name="Ollama",
        distribution="langchain-ollama",
        import_name="langchain_ollama",
        extra="ollama",
        credential_variables=(),
        credential_policy="none",
        note="The selected local model must support tool calling.",
    ),
)

PROVIDERS: Mapping[str, ProviderSpec] = MappingProxyType({item.provider: item for item in _PROVIDER_ITEMS})


def parse_model_identifier(value: str) -> ModelIdentifier:
    """Parse and canonicalize a LangChain ``provider:model`` identifier."""

    if not isinstance(value, str):
        raise TypeError("model identifier must be a string")
    if value != value.strip() or not value:
        raise ValueError("model identifier must be non-empty and have no surrounding whitespace")
    if any(character.isspace() or not character.isprintable() for character in value):
        raise ValueError("model identifier cannot contain whitespace or control characters")

    provider, separator, model = value.partition(":")
    if not separator or not model:
        raise ValueError("model identifier must use the 'provider:model' form")
    provider = _ALIASES.get(provider, provider)
    if not _PROVIDER_PATTERN.fullmatch(provider):
        raise ValueError(
            "model provider must start with a lowercase letter and contain only lowercase letters, digits, or '_'"
        )
    return ModelIdentifier(provider=provider, model=model)


def validate_agent_model(value: str) -> str:
    """Validate a model for DeepAgents and return its canonical identifier."""

    identifier = parse_model_identifier(value)
    if identifier.provider == "deepseek" and identifier.model == "deepseek-reasoner":
        raise ValueError(
            "deepseek:deepseek-reasoner cannot be used by Caro because it does not support tool calling; "
            "use deepseek:deepseek-v4-flash"
        )
    return identifier.value


def get_provider_spec(model_or_provider: str) -> ProviderSpec | None:
    """Return catalog metadata for a provider or full model identifier."""

    provider = (
        parse_model_identifier(model_or_provider).provider
        if ":" in model_or_provider
        else _ALIASES.get(model_or_provider, model_or_provider)
    )
    return PROVIDERS.get(provider)


def provider_status(
    provider: str,
    *,
    environ: Mapping[str, str] | None = None,
    version_getter: Callable[[str], str] = metadata.version,
) -> ProviderStatus:
    """Inspect integration installation and credential presence without importing it."""

    canonical_provider = _ALIASES.get(provider, provider)
    spec = PROVIDERS.get(canonical_provider)
    if spec is None:
        return ProviderStatus(
            provider=canonical_provider,
            spec=None,
            installed_version=None,
            credential_state="unmanaged",
            credential_variables=(),
        )

    try:
        installed_version = version_getter(spec.distribution)
    except metadata.PackageNotFoundError:
        installed_version = None

    environment = os.environ if environ is None else environ
    if spec.credential_policy == "none":
        credential_state: CredentialState = "local"
    elif any(environment.get(variable) for variable in spec.credential_variables):
        credential_state = "configured"
    elif spec.credential_policy == "optional":
        credential_state = "external"
    else:
        credential_state = "missing"

    return ProviderStatus(
        provider=canonical_provider,
        spec=spec,
        installed_version=installed_version,
        credential_state=credential_state,
        credential_variables=spec.credential_variables,
    )

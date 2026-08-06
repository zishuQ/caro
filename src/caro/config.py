"""Typed Caro configuration loaded from YAML and environment variables."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from caro.models.providers import validate_agent_model

GraphStreamMode = Literal["messages", "updates", "tasks", "checkpoints"]
_GRAPH_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _default_stream_modes() -> list[GraphStreamMode]:
    return ["messages", "updates"]


class ModelDefinition(BaseModel):
    """One LangChain chat model configuration."""

    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=2, ge=0, le=20)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Canonicalize and capability-check the model identifier."""

        return validate_agent_model(value)

    def init_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for ``init_chat_model``."""

        values: dict[str, Any] = {"max_retries": self.max_retries, **self.options}
        if self.temperature is not None:
            values["temperature"] = self.temperature
        if self.max_tokens is not None:
            values["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            values["timeout"] = self.timeout
        return values


class RuleMatch(BaseModel):
    """Deterministic conditions for a model-routing rule."""

    model_config = ConfigDict(extra="forbid")

    any_keywords: list[str] = Field(default_factory=list)
    regex: str | None = None
    min_prompt_chars: int | None = Field(default=None, ge=0)
    max_prompt_chars: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> RuleMatch:
        """Validate prompt-length bounds."""

        if (
            self.min_prompt_chars is not None
            and self.max_prompt_chars is not None
            and self.min_prompt_chars > self.max_prompt_chars
        ):
            raise ValueError("min_prompt_chars cannot exceed max_prompt_chars")
        return self


class RoutingRule(BaseModel):
    """Select a named model when all configured conditions match."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    match: RuleMatch = Field(default_factory=RuleMatch)


class RoutingSettings(BaseModel):
    """Model pool, ordered routing rules, and failure fallbacks."""

    model_config = ConfigDict(extra="forbid")

    default: str = "primary"
    models: dict[str, ModelDefinition] = Field(
        default_factory=lambda: {
            "primary": ModelDefinition(model="openai:gpt-5.6-sol", temperature=0.0),
        }
    )
    rules: list[RoutingRule] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> RoutingSettings:
        """Ensure every model reference targets the declared pool."""

        referenced = {self.default, *self.fallbacks, *(rule.model for rule in self.rules)}
        missing = sorted(referenced.difference(self.models))
        if missing:
            raise ValueError(f"routing references unknown model(s): {', '.join(missing)}")
        if len(self.fallbacks) != len(set(self.fallbacks)):
            raise ValueError("routing.fallbacks cannot contain duplicates")
        if self.default in self.fallbacks:
            raise ValueError("routing.default cannot also appear in routing.fallbacks")
        return self


class ApprovalSettings(BaseModel):
    """Tool approval policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    minimum_risk: Literal["low", "medium", "high", "critical"] = "high"


class WorkspaceSettings(BaseModel):
    """Workspace and local-shell restrictions."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Path(".")
    shell_enabled: bool = True
    sandbox_backend: Literal["local", "bubblewrap"] = "local"
    shell_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    inherit_environment: bool = False
    allowed_environment: list[str] = Field(default_factory=lambda: ["PATH", "LANG", "LC_ALL", "TERM"])


class NetworkSettings(BaseModel):
    """Optional process-wide proxy settings for provider HTTP clients.

    LangChain provider integrations use standard proxy environment variables.
    These fields provide a project-local alternative for users who do not want
    to export variables in their shell. Explicit environment values always win.
    """

    model_config = ConfigDict(extra="forbid")

    http_proxy: str | None = None
    https_proxy: str | None = None
    all_proxy: str | None = None
    no_proxy: str | None = None
    openai_proxy: str | None = None

    @field_validator(
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "openai_proxy",
        mode="before",
    )
    @classmethod
    def validate_proxy_url(cls, value: object) -> str | None:
        """Require a proxy URL with a supported scheme and authority."""

        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise TypeError("proxy values must be strings")
        candidate = value.strip()
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https", "socks5", "socks5h"} or not parsed.netloc:
            raise ValueError(
                "proxy must be an http(s):// or socks5(h):// URL with a host"
            )
        if parsed.username is not None or parsed.password is not None:
            # Credentials are supported by HTTPX, but remain in the value so
            # they are never printed by diagnostics or included in logs.
            return candidate
        return candidate

    @field_validator("no_proxy", mode="before")
    @classmethod
    def normalize_no_proxy(cls, value: object) -> str | None:
        """Normalize an optional comma-separated bypass list."""

        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise TypeError("no_proxy must be a string")
        return value.strip() or None


class StateSettings(BaseModel):
    """SQLite persistence paths."""

    model_config = ConfigDict(extra="forbid")

    directory: Path = Path(".caro")
    runs_database: str = "runs.sqlite3"
    checkpoints_database: str = "checkpoints.sqlite3"


class MemorySettings(BaseModel):
    """Project-scoped file memory plus governed structured long-term memory."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sources: list[str] = Field(
        default_factory=lambda: [".caro/memory/AGENTS.md"],
        min_length=1,
        max_length=16,
    )
    initialize_missing: bool = True
    max_file_bytes: int = Field(default=131_072, ge=1024, le=1_048_576)
    structured_enabled: bool = True
    structured_database: str = "memory.sqlite3"
    auto_retrieve: bool = True
    retrieval_limit: int = Field(default=8, ge=1, le=50)
    max_injected_chars: int = Field(default=12_000, ge=500, le=100_000)
    minimum_importance: float = Field(default=0.2, ge=0, le=1)
    minimum_confidence: float = Field(default=0.6, ge=0, le=1)
    max_item_chars: int = Field(default=4000, ge=100, le=20_000)
    episodic_ttl_days: int = Field(default=90, ge=1, le=3650)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, values: list[str]) -> list[str]:
        """Require normalized POSIX paths relative to the workspace."""

        seen: set[str] = set()
        for value in values:
            parts = value.split("/")
            if (
                not value
                or value.startswith(("/", "~"))
                or "\\" in value
                or any(part in {"", ".", ".."} for part in parts)
            ):
                raise ValueError(f"memory source must be a normalized workspace-relative POSIX path: {value!r}")
            if value in seen:
                raise ValueError(f"duplicate memory source: {value}")
            seen.add(value)
        return values

    @field_validator("structured_database")
    @classmethod
    def validate_structured_database(cls, value: str) -> str:
        """Keep the structured database name beneath the configured state directory."""

        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or not value.casefold().endswith((".sqlite", ".sqlite3", ".db"))
        ):
            raise ValueError("structured_database must be a plain SQLite filename")
        return value


class AgentDelegateSettings(BaseModel):
    """One explicitly allowed subagent exposed through a role's ``task`` tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(min_length=1, max_length=500)
    system_prompt: str = Field(min_length=1)
    model: str = Field(min_length=1)


class AgentRoleSettings(BaseModel):
    """One deterministic orchestration role backed by a DeepAgent subgraph."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=500)
    system_prompt: str = Field(min_length=1)
    model: str | None = None
    memory: bool = True
    shell_enabled: bool | None = None
    delegates: list[AgentDelegateSettings] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_delegates(self) -> AgentRoleSettings:
        """Reject ambiguous or reserved delegate names within one role."""

        names = [delegate.name for delegate in self.delegates]
        if len(names) != len(set(names)):
            raise ValueError("an orchestration role cannot contain duplicate delegates")
        if any(name == "general-purpose" for name in names):
            raise ValueError("delegate name 'general-purpose' is reserved by DeepAgents")
        return self


class OrchestrationStageSettings(BaseModel):
    """One sequential stage containing one or more parallel agents."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    agents: list[str] = Field(min_length=1, max_length=16)

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, values: list[str]) -> list[str]:
        """Reject duplicate or unsafe role names within a stage."""

        if len(values) != len(set(values)):
            raise ValueError("an orchestration stage cannot contain duplicate agents")
        invalid = [value for value in values if _GRAPH_NAME_PATTERN.fullmatch(value) is None]
        if invalid:
            raise ValueError(f"invalid orchestration agent name(s): {', '.join(invalid)}")
        return values


class OrchestrationSettings(BaseModel):
    """Deterministic LangGraph fan-out/fan-in orchestration configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    agents: dict[str, AgentRoleSettings] = Field(default_factory=dict)
    stages: list[OrchestrationStageSettings] = Field(default_factory=list, max_length=32)
    max_parallel_agents: int = Field(default=4, ge=1, le=16)
    allow_nested_delegation: bool = False
    max_report_chars: int = Field(default=20_000, ge=1000, le=100_000)
    max_handoff_chars: int = Field(default=40_000, ge=2000, le=200_000)

    @model_validator(mode="after")
    def validate_graph(self) -> OrchestrationSettings:
        """Validate role references and bounded stage parallelism."""

        if self.enabled and (not self.agents or not self.stages):
            raise ValueError("enabled orchestration requires at least one agent and one stage")
        invalid_names = [name for name in self.agents if _GRAPH_NAME_PATTERN.fullmatch(name) is None]
        if invalid_names:
            raise ValueError(f"invalid orchestration agent name(s): {', '.join(invalid_names)}")
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("orchestration stage names must be unique")
        referenced = {agent for stage in self.stages for agent in stage.agents}
        missing = sorted(referenced.difference(self.agents))
        if missing:
            raise ValueError(f"orchestration stages reference unknown agent(s): {', '.join(missing)}")
        unused = sorted(set(self.agents).difference(referenced))
        if self.enabled and unused:
            raise ValueError(f"enabled orchestration contains unused agent(s): {', '.join(unused)}")
        oversized = [stage.name for stage in self.stages if len(stage.agents) > self.max_parallel_agents]
        if oversized:
            raise ValueError(
                "orchestration stage exceeds max_parallel_agents: " + ", ".join(oversized)
            )
        delegated_roles = sorted(
            name for name, role in self.agents.items() if role.delegates
        )
        if delegated_roles and not self.allow_nested_delegation:
            raise ValueError(
                "orchestration roles with delegates require allow_nested_delegation=true: "
                + ", ".join(delegated_roles)
            )
        return self


class LangfuseSettings(BaseModel):
    """Langfuse callback configuration; credentials stay in the environment."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    environment: str = "local"
    tags: list[str] = Field(default_factory=lambda: ["caro"])


class GraphSettings(BaseModel):
    """LangGraph runtime behavior exposed by Caro."""

    model_config = ConfigDict(extra="forbid")

    recursion_limit: int = Field(default=9999, ge=1)
    stream_modes: list[GraphStreamMode] = Field(default_factory=_default_stream_modes)
    history_limit: int = Field(default=50, ge=1, le=1000)


class CodeIntegrationSettings(BaseModel):
    """Compatibility surface for the maintained DeepAgents Code runtime.

    Caro owns the governed graph and persistence, while this section controls
    which existing terminal-agent integrations are attached to the interactive
    Textual frontend.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    skills_enabled: bool = True
    mcp_enabled: bool = True
    mcp_config: Path | None = None
    trust_project_mcp: bool | None = None
    ask_user_enabled: bool = True
    interpreter_enabled: bool = False


class CaroSettings(BaseSettings):
    """Top-level Caro settings."""

    model_config = SettingsConfigDict(
        env_prefix="CARO_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
    )

    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    approvals: ApprovalSettings = Field(default_factory=ApprovalSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    state: StateSettings = Field(default_factory=StateSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    orchestration: OrchestrationSettings = Field(default_factory=OrchestrationSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    code: CodeIntegrationSettings = Field(default_factory=CodeIntegrationSettings)
    system_prompt: str = "You are Caro, a governed terminal coding agent. Work only inside the configured workspace."

    @model_validator(mode="after")
    def validate_orchestration_models(self) -> CaroSettings:
        """Ensure role models exist and roles cannot elevate global Shell access."""

        missing_models = sorted(
            {
                role.model
                for role in self.orchestration.agents.values()
                if role.model is not None and role.model not in self.routing.models
            }
        )
        missing_delegate_models = sorted(
            {
                delegate.model
                for role in self.orchestration.agents.values()
                for delegate in role.delegates
                if delegate.model not in self.routing.models
            }
        )
        if missing_models:
            raise ValueError(f"orchestration roles reference unknown model(s): {', '.join(missing_models)}")
        if missing_delegate_models:
            raise ValueError(
                "orchestration delegates reference unknown model(s): "
                + ", ".join(missing_delegate_models)
            )
        if not self.workspace.shell_enabled:
            elevated = sorted(
                name for name, role in self.orchestration.agents.items() if role.shell_enabled is True
            )
            if elevated:
                raise ValueError(
                    "orchestration roles cannot enable Shell when workspace.shell_enabled is false: "
                    + ", ".join(elevated)
                )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let explicit ``CARO_*`` environment variables override YAML values."""

        del settings_cls
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    def fingerprint(self) -> str:
        """Return a stable configuration fingerprint for run provenance."""

        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_settings(path: Path | None = None, *, workspace: Path | None = None) -> CaroSettings:
    """Load settings from YAML, ``.env``, environment, and CLI overrides."""

    env_candidates: list[Path] = []
    if path is not None:
        env_candidates.append(path.expanduser().resolve().parent / ".env")
    if workspace is not None:
        workspace_path = workspace.expanduser().resolve()
        env_candidates.append(workspace_path / ".env")
    env_candidates.append(Path.cwd() / ".env")
    seen_env_paths: set[Path] = set()
    for candidate in env_candidates:
        resolved_env = candidate.expanduser().resolve()
        if resolved_env in seen_env_paths:
            continue
        seen_env_paths.add(resolved_env)
        if resolved_env.is_file():
            load_env(resolved_env)
            break

    data: dict[str, Any] = {}
    if path is not None:
        resolved_path = path.expanduser().resolve()
        loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("Caro YAML root must be a mapping")
        data = {} if loaded is None else loaded

    if workspace is not None:
        workspace_values = dict(data.get("workspace", {}))
        workspace_values["root"] = workspace
        data["workspace"] = workspace_values

    settings = CaroSettings(**data)
    apply_network_environment(settings.network)
    settings.workspace.root = settings.workspace.root.expanduser().resolve()
    state_directory = settings.state.directory.expanduser()
    if not state_directory.is_absolute():
        state_directory = settings.workspace.root / state_directory
    settings.state.directory = state_directory.resolve()
    return settings


_PROXY_ENV_FIELDS: tuple[tuple[str, str], ...] = (
    ("http_proxy", "HTTP_PROXY"),
    ("https_proxy", "HTTPS_PROXY"),
    ("all_proxy", "ALL_PROXY"),
    ("no_proxy", "NO_PROXY"),
    ("openai_proxy", "OPENAI_PROXY"),
)


def apply_network_environment(network: NetworkSettings) -> None:
    """Apply configured proxy values without overriding explicit environment.

    Args:
        network: Project network settings.

    Environment variables are intentionally mutated once at startup so all
    LangChain integrations, including providers Caro does not know about,
    share the same HTTPX proxy behavior. A value already present in the
    environment, including an explicit empty value, remains authoritative.
    """

    for field_name, env_name in _PROXY_ENV_FIELDS:
        value = getattr(network, field_name)
        if (
            value is not None
            and env_name not in os.environ
            and env_name.lower() not in os.environ
        ):
            os.environ[env_name] = value


def load_env(path: Path | None = None) -> Path | None:
    """Load a local dotenv file without overriding explicit environment variables.

    Args:
        path: Dotenv file to load. Defaults to ``.env`` in the current directory.

    Returns:
        The resolved dotenv path when a file was loaded, otherwise ``None``.
    """

    candidate = Path.cwd() / ".env" if path is None else path.expanduser()
    resolved = candidate.resolve()
    if not resolved.is_file():
        return None
    load_dotenv(dotenv_path=resolved, override=False)
    return resolved

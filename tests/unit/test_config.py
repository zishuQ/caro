"""Pydantic Settings and YAML tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from caro.cli import _proxy_status
from caro.config import (
    CaroSettings,
    ModelDefinition,
    apply_network_environment,
    load_env,
    load_settings,
)


def test_model_definition_canonicalizes_google_alias() -> None:
    definition = ModelDefinition(model="gemini:gemini-2.5-pro")
    assert definition.model == "google_genai:gemini-2.5-pro"


def test_routing_references_must_exist() -> None:
    with pytest.raises(ValidationError, match="unknown model"):
        CaroSettings(routing={"default": "missing", "models": {"primary": {"model": "openai:gpt-5.6-sol"}}})


def test_yaml_and_workspace_override_are_resolved(tmp_path: Path) -> None:
    config = tmp_path / "caro.yaml"
    config.write_text(
        "routing:\n"
        "  default: ds\n"
        "  models:\n"
        "    ds:\n"
        "      model: deepseek:deepseek-v4-flash\n"
        "state:\n"
        "  directory: .state\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "repo"
    workspace.mkdir()

    settings = load_settings(config, workspace=workspace)

    assert settings.routing.models["ds"].model == "deepseek:deepseek-v4-flash"
    assert settings.workspace.root == workspace.resolve()
    assert settings.state.directory == (workspace / ".state").resolve()
    assert len(settings.fingerprint()) == 64


def test_load_env_reads_dotenv_without_overriding_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=from-file\nCARO_APPROVALS__ENABLED=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("CARO_APPROVALS__ENABLED", raising=False)

    assert load_env(env_file) == env_file.resolve()
    assert os.environ["DEEPSEEK_API_KEY"] == "from-file"
    assert os.environ["CARO_APPROVALS__ENABLED"] == "false"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "explicit")
    load_env(env_file)
    assert os.environ["DEEPSEEK_API_KEY"] == "explicit"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("CARO_APPROVALS__ENABLED", raising=False)


def test_load_settings_automatically_loads_adjacent_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "caro.yaml"
    config.write_text("approvals:\n  enabled: true\n", encoding="utf-8")
    (tmp_path / ".env").write_text("CARO_APPROVALS__ENABLED=false\n", encoding="utf-8")
    monkeypatch.delenv("CARO_APPROVALS__ENABLED", raising=False)

    settings = load_settings(config)

    assert settings.approvals.enabled is False
    monkeypatch.delenv("CARO_APPROVALS__ENABLED", raising=False)


def test_workspace_sandbox_backend_is_strict() -> None:
    settings = CaroSettings(workspace={"sandbox_backend": "bubblewrap"})
    assert settings.workspace.sandbox_backend == "bubblewrap"
    with pytest.raises(ValidationError):
        CaroSettings(workspace={"sandbox_backend": "container"})


def test_network_proxy_settings_apply_without_overriding_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = CaroSettings(
        network={
            "http_proxy": "http://proxy.example:8080",
            "https_proxy": "socks5://proxy.example:1080",
            "no_proxy": "localhost,127.0.0.1",
        }
    )
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://shell.example:3128")
    monkeypatch.setenv("https_proxy", "http://shell.example:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    apply_network_environment(settings.network)

    assert os.environ["HTTP_PROXY"] == "http://proxy.example:8080"
    assert os.environ["HTTPS_PROXY"] == "http://shell.example:3128"
    assert os.environ["NO_PROXY"] == "localhost,127.0.0.1"


@pytest.mark.parametrize(
    "value",
    ["proxy.example:8080", "ftp://proxy.example:8080", "http://"],
)
def test_network_proxy_requires_supported_url(value: str) -> None:
    with pytest.raises(ValidationError, match="proxy"):
        CaroSettings(network={"http_proxy": value})


def test_yaml_network_proxy_is_applied_before_runtime_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "caro.yaml"
    config.write_text(
        "network:\n"
        "  https_proxy: http://proxy.example:7890\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)

    settings = load_settings(config)

    assert settings.network.https_proxy == "http://proxy.example:7890"
    assert os.environ["HTTPS_PROXY"] == "http://proxy.example:7890"


def test_proxy_status_redacts_proxy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@proxy.example:7890")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("OPENAI_PROXY", raising=False)
    monkeypatch.delenv("openai_proxy", raising=False)

    status = _proxy_status()

    assert "proxy.example:7890" in status
    assert "secret" not in status
    assert "user" not in status


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "caro.yaml"
    config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(config)


def test_memory_sources_are_strict_and_affect_fingerprint() -> None:
    default = CaroSettings()
    custom = CaroSettings(memory={"sources": ["docs/AGENTS.md"]})

    assert default.memory.sources == [".caro/memory/AGENTS.md"]
    assert default.fingerprint() != custom.fingerprint()

    with pytest.raises(ValidationError, match="duplicate memory source"):
        CaroSettings(memory={"sources": ["AGENTS.md", "AGENTS.md"]})


def test_structured_memory_settings_are_bounded_and_strict() -> None:
    settings = CaroSettings(
        memory={
            "structured_enabled": True,
            "structured_database": "project-memory.sqlite3",
            "retrieval_limit": 12,
            "max_injected_chars": 24000,
            "minimum_importance": 0.3,
            "minimum_confidence": 0.75,
            "max_item_chars": 3000,
            "episodic_ttl_days": 45,
        }
    )

    assert settings.memory.structured_database == "project-memory.sqlite3"
    assert settings.memory.retrieval_limit == 12
    with pytest.raises(ValidationError, match="plain SQLite filename"):
        CaroSettings(memory={"structured_database": "../memory.sqlite3"})
    with pytest.raises(ValidationError):
        CaroSettings(memory={"minimum_confidence": 1.1})


def test_code_integration_defaults_and_strict_mcp_config() -> None:
    settings = CaroSettings(
        code={
            "enabled": True,
            "skills_enabled": False,
            "mcp_enabled": True,
            "mcp_config": ".deepagents/mcp.json",
            "trust_project_mcp": False,
        }
    )

    assert settings.code.enabled is True
    assert settings.code.skills_enabled is False
    assert settings.code.mcp_config == Path(".deepagents/mcp.json")
    with pytest.raises(ValidationError):
        CaroSettings(code={"unexpected": True})


def test_orchestration_configuration_is_strict() -> None:
    with pytest.raises(ValidationError, match="requires at least one agent"):
        CaroSettings(orchestration={"enabled": True})

    with pytest.raises(ValidationError, match="unknown agent"):
        CaroSettings(
            orchestration={
                "enabled": True,
                "agents": {"reviewer": {"description": "Review", "system_prompt": "Review."}},
                "stages": [{"name": "review", "agents": ["missing"]}],
            }
        )

    with pytest.raises(ValidationError, match="unknown model"):
        CaroSettings(
            orchestration={
                "enabled": True,
                "agents": {
                    "reviewer": {
                        "description": "Review",
                        "system_prompt": "Review.",
                        "model": "missing-model",
                    }
                },
                "stages": [{"name": "review", "agents": ["reviewer"]}],
            }
        )


def test_orchestration_cannot_elevate_shell_or_exceed_parallel_limit() -> None:
    agent = {"description": "Review", "system_prompt": "Review.", "shell_enabled": True}
    with pytest.raises(ValidationError, match="cannot enable Shell"):
        CaroSettings(
            workspace={"shell_enabled": False},
            orchestration={
                "enabled": True,
                "agents": {"reviewer": agent},
                "stages": [{"name": "review", "agents": ["reviewer"]}],
            },
        )
    with pytest.raises(ValidationError, match="exceeds max_parallel_agents"):
        CaroSettings(
            orchestration={
                "enabled": True,
                "max_parallel_agents": 1,
                "agents": {
                    "reviewer": {"description": "Review", "system_prompt": "Review."},
                    "tester": {"description": "Test", "system_prompt": "Test."},
                },
                "stages": [{"name": "verify", "agents": ["reviewer", "tester"]}],
            }
        )


def test_orchestration_delegate_configuration_is_explicit_and_allowlisted() -> None:
    settings = CaroSettings(
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
                            "description": "Gather focused repository evidence.",
                            "system_prompt": "Return evidence only.",
                            "model": "cheap",
                        }
                    ],
                }
            },
            "stages": [{"name": "analysis", "agents": ["architect"]}],
        },
    )

    delegate = settings.orchestration.agents["architect"].delegates[0]
    assert delegate.name == "cheap_researcher"
    assert delegate.model == "cheap"


def test_orchestration_delegate_references_and_names_are_validated() -> None:
    common = {
        "enabled": True,
        "agents": {
            "architect": {
                "description": "Design the change.",
                "system_prompt": "Analyze the architecture.",
                "delegates": [
                    {
                        "name": "cheap_researcher",
                        "description": "Gather evidence.",
                        "system_prompt": "Return evidence.",
                        "model": "missing",
                    }
                ],
            }
        },
        "stages": [{"name": "analysis", "agents": ["architect"]}],
    }
    with pytest.raises(ValidationError, match="allow_nested_delegation"):
        CaroSettings(orchestration=common)

    common["allow_nested_delegation"] = True
    with pytest.raises(ValidationError, match="unknown model"):
        CaroSettings(orchestration=common)

    common["agents"]["architect"]["delegates"] = [
        {
            "name": "cheap_researcher",
            "description": "Gather evidence.",
            "system_prompt": "Return evidence.",
            "model": "primary",
        },
        {
            "name": "cheap_researcher",
            "description": "Gather more evidence.",
            "system_prompt": "Return more evidence.",
            "model": "primary",
        },
    ]
    with pytest.raises(ValidationError, match="duplicate delegates"):
        CaroSettings(orchestration=common)

    common["agents"]["architect"]["delegates"][1]["name"] = "general-purpose"
    with pytest.raises(ValidationError, match="reserved"):
        CaroSettings(orchestration=common)

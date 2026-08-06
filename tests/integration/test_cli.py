"""Terminal command integration tests that do not call provider APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from caro.cli import app

runner = CliRunner()


def write_config(path: Path, provider_model: str = "deepseek:deepseek-v4-flash") -> Path:
    config = path / "caro.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "routing": {"default": "primary", "models": {"primary": {"model": provider_model}}},
                "workspace": {"root": str(path)},
                "state": {"directory": str(path / ".caro")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


def test_help_and_version_are_branded_caro() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["version"])
    assert help_result.exit_code == 0
    assert "Caro" in help_result.output
    assert "graph" in help_result.output
    assert "agents" in help_result.output
    assert version_result.output.strip() == "caro 0.5.0"


def test_providers_and_tools_show_governed_configuration(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    providers_result = runner.invoke(
        app,
        ["--config", str(config), "providers"],
        env={"DEEPSEEK_API_KEY": "test-key", "NO_COLOR": "1"},
    )
    tools_result = runner.invoke(app, ["--config", str(config), "tools"], env={"NO_COLOR": "1"})
    assert providers_result.exit_code == 0
    assert "DeepSeek" in providers_result.output
    assert "1.1.0" in providers_result.output
    assert tools_result.exit_code == 0
    assert "execute" in tools_result.output
    assert "critical" in tools_result.output


def test_local_config_is_discovered_without_config_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config(tmp_path)
    config.rename(tmp_path / "caro.local.yaml")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["providers"],
        env={"DEEPSEEK_API_KEY": "test-key", "NO_COLOR": "1"},
    )

    assert result.exit_code == 0
    assert "DeepSeek" in result.output


def test_agents_command_shows_single_and_orchestrated_modes(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    single = runner.invoke(app, ["--config", str(config), "agents"], env={"NO_COLOR": "1"})
    orchestrated = runner.invoke(
        app,
        ["--config", "examples/caro/multi-agent.yaml", "agents"],
        env={"NO_COLOR": "1"},
    )

    assert single.exit_code == 0
    assert "general-purpose" in single.output
    assert orchestrated.exit_code == 0
    assert "analysis" in orchestrated.output
    assert "parallel" in orchestrated.output
    assert "investigator" in orchestrated.output
    assert "synthesizer" in orchestrated.output
    assert "focused_researcher" in orchestrated.output


def test_doctor_fails_for_missing_provider_key_and_passes_when_set(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    failed = runner.invoke(
        app,
        ["--config", str(config), "doctor"],
        # Keep this scenario independent from the developer's automatically loaded project `.env`.
        env={"DEEPSEEK_API_KEY": "", "NO_COLOR": "1"},
    )
    passed = runner.invoke(
        app,
        ["--config", str(config), "doctor"],
        env={"DEEPSEEK_API_KEY": "test-key", "NO_COLOR": "1"},
    )
    assert failed.exit_code == 1
    assert "DEEPSEEK_API_KEY" in failed.output
    assert passed.exit_code == 0


def test_init_creates_provider_specific_yaml(tmp_path: Path) -> None:
    output = tmp_path / "generated.yaml"
    result = runner.invoke(app, ["init", "--provider", "deepseek", "--output", str(output)])
    assert result.exit_code == 0
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["routing"]["models"]["primary"]["model"] == "deepseek:deepseek-v4-flash"
    assert payload["memory"]["sources"] == [".caro/memory/AGENTS.md"]
    assert payload["orchestration"] == {"enabled": False}


def test_memory_cli_initializes_adds_and_shows_project_memory(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    status_before = runner.invoke(app, ["--config", str(config), "memory", "status"], env={"NO_COLOR": "1"})
    initialized = runner.invoke(app, ["--config", str(config), "memory", "init"], env={"NO_COLOR": "1"})
    added = runner.invoke(
        app,
        ["--config", str(config), "memory", "add", "Use pytest -q before commits."],
        env={"NO_COLOR": "1"},
    )
    shown = runner.invoke(app, ["--config", str(config), "memory", "show"], env={"NO_COLOR": "1"})

    assert status_before.exit_code == 0
    assert "not initialized" in status_before.output
    assert initialized.exit_code == 0
    assert added.exit_code == 0
    assert shown.exit_code == 0
    assert "Use pytest -q before commits." in shown.output
    assert (tmp_path / ".caro" / "memory" / "AGENTS.md").is_file()


def test_structured_memory_cli_consolidates_searches_audits_and_archives(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    remembered = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "memory",
            "remember",
            "Run uv run pytest -q before committing parser changes.",
            "--kind",
            "procedural",
            "--importance",
            "0.9",
            "--confidence",
            "1.0",
            "--evidence",
            "Operator verified the command.",
        ],
        env={"NO_COLOR": "1"},
    )
    searched = runner.invoke(
        app,
        ["--config", str(config), "memory", "search", "pytest parser"],
        env={"NO_COLOR": "1"},
    )
    stats = runner.invoke(app, ["--config", str(config), "memory", "stats"], env={"NO_COLOR": "1"})
    history = runner.invoke(app, ["--config", str(config), "memory", "history"], env={"NO_COLOR": "1"})

    assert remembered.exit_code == 0, remembered.output
    assert "created" in remembered.output
    assert searched.exit_code == 0
    assert "pytest -q" in searched.output
    assert stats.exit_code == 0
    assert "active" in stats.output and "1" in stats.output
    assert history.exit_code == 0
    assert "operator" in history.output
    assert (tmp_path / ".caro" / "memory.sqlite3").is_file()

    memory_id = remembered.output.split("memory_id: ", 1)[1].splitlines()[0].strip()
    archived = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "memory",
            "archive",
            memory_id,
            "--reason",
            "Replaced by a new verified workflow.",
        ],
        env={"NO_COLOR": "1"},
    )
    after = runner.invoke(
        app,
        ["--config", str(config), "memory", "search", "pytest parser"],
        env={"NO_COLOR": "1"},
    )

    assert archived.exit_code == 0
    assert "archived" in archived.output
    assert "pytest -q" not in after.output


def test_structured_memory_cli_rejects_secret_without_persisting_content(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    rejected = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "memory",
            "remember",
            "API_KEY=sk-secret-material-1234567890",
            "--kind",
            "semantic",
            "--importance",
            "1",
            "--confidence",
            "1",
            "--evidence",
            "operator input",
        ],
        env={"NO_COLOR": "1"},
    )
    searched = runner.invoke(
        app,
        ["--config", str(config), "memory", "search", "secret material"],
        env={"NO_COLOR": "1"},
    )

    assert rejected.exit_code == 1
    assert "credential-like" in rejected.output
    assert "secret-material" not in searched.output


def test_memory_benchmark_cli_is_offline_and_machine_readable(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    seeded = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "memory",
            "remember",
            "The parser entry point is src/parser.py.",
            "--kind",
            "semantic",
            "--evidence",
            "Verified in the repository.",
        ],
        env={"NO_COLOR": "1"},
    )
    dataset = tmp_path / "benchmark.yaml"
    dataset.write_text(
        "cases:\n"
        "  - name: parser-location\n"
        "    query: parser entry point\n"
        "    scopes: [project]\n"
        "    expected_substrings: [src/parser.py]\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["--config", str(config), "memory", "benchmark", str(dataset), "--json"],
        env={"NO_COLOR": "1"},
    )

    assert seeded.exit_code == 0
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["hit_rate"] == 1.0
    assert payload["recall_at_k"] == 1.0


def test_graph_fork_update_parser_is_strict() -> None:
    from caro.cli import _parse_updates

    assert _parse_updates(["count=2", 'topic="chickens"']) == {"count": 2, "topic": "chickens"}

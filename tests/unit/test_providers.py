"""Provider catalog tests."""

from __future__ import annotations

from importlib import metadata

import pytest

from caro.models.providers import PROVIDERS, parse_model_identifier, provider_status, validate_agent_model


def test_catalog_covers_first_party_adapters() -> None:
    assert set(PROVIDERS) == {"openai", "deepseek", "anthropic", "google_genai", "openrouter", "ollama"}
    assert PROVIDERS["deepseek"].distribution == "langchain-deepseek"
    assert PROVIDERS["openai"].credential_variables == ("OPENAI_API_KEY",)


def test_parse_model_splits_only_first_colon() -> None:
    identifier = parse_model_identifier("ollama:gpt-oss:20b")
    assert identifier.provider == "ollama"
    assert identifier.model == "gpt-oss:20b"


@pytest.mark.parametrize("alias", ["google:gemini-2.5-pro", "gemini:gemini-2.5-pro"])
def test_google_aliases_are_canonicalized(alias: str) -> None:
    assert parse_model_identifier(alias).value == "google_genai:gemini-2.5-pro"


@pytest.mark.parametrize("value", ["gpt-5", "openai:", ":gpt-5", "OpenAI:gpt-5", "openai:bad model", " openai:gpt-5"])
def test_invalid_model_identifiers_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        parse_model_identifier(value)


def test_unknown_provider_remains_compatible() -> None:
    assert validate_agent_model("groq:llama-3.3-70b") == "groq:llama-3.3-70b"
    assert provider_status("groq", environ={}).ready is None


def test_deepseek_reasoner_is_rejected_for_agent_tool_calling() -> None:
    with pytest.raises(ValueError, match="deepseek-v4-flash"):
        validate_agent_model("deepseek:deepseek-reasoner")


def test_required_credentials_and_package_are_checked() -> None:
    status = provider_status("deepseek", environ={"DEEPSEEK_API_KEY": "set"}, version_getter=lambda _: "1.1.0")
    assert status.ready is True
    assert status.installed_version == "1.1.0"
    assert status.credential_state == "configured"


def test_missing_distribution_is_not_ready() -> None:
    def missing(_: str) -> str:
        raise metadata.PackageNotFoundError

    status = provider_status("openai", environ={"OPENAI_API_KEY": "set"}, version_getter=missing)
    assert status.ready is False
    assert status.installed is False


def test_google_adc_and_ollama_are_not_false_failures() -> None:
    google = provider_status("google_genai", environ={}, version_getter=lambda _: "4.3.2")
    ollama = provider_status("ollama", environ={}, version_getter=lambda _: "1.1.0")
    assert google.credential_state == "external"
    assert google.ready is None
    assert ollama.credential_state == "local"
    assert ollama.ready is True

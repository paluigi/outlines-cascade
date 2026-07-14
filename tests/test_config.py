"""Tests for config parsing and provider registry."""

import pytest

from outlines_cascade.config import (
    CascadeEntry,
    ProviderKind,
    load_config,
    provider_kind,
)


class TestProviderKind:
    def test_cloud_providers(self):
        assert provider_kind("openai") == ProviderKind.CLOUD
        assert provider_kind("anthropic") == ProviderKind.CLOUD
        assert provider_kind("gemini") == ProviderKind.CLOUD
        assert provider_kind("ollama") == ProviderKind.CLOUD

    def test_steerable_providers(self):
        assert provider_kind("transformers") == ProviderKind.STEERABLE
        assert provider_kind("llamacpp") == ProviderKind.STEERABLE

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider type"):
            provider_kind("unknown_provider")


class TestCascadeEntry:
    def test_basic(self):
        e = CascadeEntry(provider="openai", model="gpt-4o")
        assert e.provider == "openai"
        assert e.model == "gpt-4o"
        assert e.supported_types is None
        assert e.provider_kind is None

    def test_with_supported_types(self):
        e = CascadeEntry(
            provider="local",
            model="phi-3",
            supported_types=["json", "regex"],
        )
        assert e.supported_types == ["json", "regex"]

    def test_with_provider_kind(self):
        e = CascadeEntry(
            provider="local",
            model="phi-3",
            provider_kind=ProviderKind.STEERABLE,
        )
        assert e.provider_kind == ProviderKind.STEERABLE


class TestLoadConfig:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")

    def test_valid_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
type = "transformers"

[cascades.primary]
entries = [
    { provider = "openai", model = "gpt-4o" },
    { provider = "local", model = "microsoft/Phi-3-mini-4k-instruct" },
]
            """
        )
        config = load_config(config_file)

        assert "openai" in config.providers
        assert config.providers["openai"].type == "openai"
        assert config.providers["local"].type == "transformers"

        assert "primary" in config.cascades
        entries = config.cascades["primary"].entries
        assert len(entries) == 2
        assert entries[0].provider == "openai"
        assert entries[0].model == "gpt-4o"
        assert entries[1].provider == "local"
        assert entries[1].model == "microsoft/Phi-3-mini-4k-instruct"

    def test_config_with_supported_types(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[providers.local]
type = "transformers"

[cascades.regex_capable]
entries = [
    { provider = "local", model = "phi-3", supported_types = ["regex", "cfg", "json"] },
]
            """
        )
        config = load_config(config_file)
        entry = config.cascades["regex_capable"].entries[0]
        assert entry.supported_types == ["regex", "cfg", "json"]

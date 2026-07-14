"""Tests for the adapter factory and adapter classes."""

import pytest

from outlines_cascade.adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    LlamaCppAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    TransformersAdapter,
    build_adapter,
)


class TestBuildAdapter:
    def test_openai_adapter(self):
        a = build_adapter(
            provider_type="openai",
            provider_name="my-openai",
            model="gpt-4o",
            api_key="test-key",
        )
        assert isinstance(a, OpenAIAdapter)
        assert a.provider == "my-openai"
        assert a.model == "gpt-4o"
        assert a.entry_key == "my-openai/gpt-4o"

    def test_anthropic_adapter(self):
        a = build_adapter(
            provider_type="anthropic",
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        assert isinstance(a, AnthropicAdapter)

    def test_gemini_adapter(self):
        a = build_adapter(
            provider_type="gemini",
            provider_name="gemini",
            model="gemini-2.0-flash",
            api_key="test-key",
        )
        assert isinstance(a, GeminiAdapter)

    def test_ollama_adapter(self):
        a = build_adapter(
            provider_type="ollama",
            provider_name="ollama",
            model="llama3.1",
        )
        assert isinstance(a, OllamaAdapter)
        assert a.provider == "ollama"

    def test_transformers_adapter(self):
        a = build_adapter(
            provider_type="transformers",
            provider_name="local",
            model="microsoft/Phi-3-mini-4k-instruct",
            device="cpu",
        )
        assert isinstance(a, TransformersAdapter)
        assert a._device == "cpu"

    def test_llamacpp_adapter(self):
        a = build_adapter(
            provider_type="llamacpp",
            provider_name="local",
            model="/models/llama.gguf",
        )
        assert isinstance(a, LlamaCppAdapter)

    def test_unknown_provider_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider type"):
            build_adapter(
                provider_type="unknown",
                provider_name="test",
                model="test",
            )

    def test_cloud_adapter_without_api_key_raises(self):
        with pytest.raises(ValueError, match="API key required"):
            build_adapter(
                provider_type="openai",
                provider_name="test",
                model="gpt-4o",
                api_key=None,
            )

    def test_base_url_passed_through(self):
        a = build_adapter(
            provider_type="openai",
            provider_name="custom",
            model="gpt-4o",
            api_key="k",
            base_url="https://custom.api.com/v1",
        )
        assert a._base_url == "https://custom.api.com/v1"

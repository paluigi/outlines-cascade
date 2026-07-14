"""Tests for the cascade engine using mock adapters."""

from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from outlines_cascade.adapters import AdapterResult, OutlinesModelAdapter
from outlines_cascade.config import CascadeEntry
from outlines_cascade.engine import StructuredCascade
from outlines_cascade.errors import (
    AllProvidersExhaustedError,
    TypeCompatibilityError,
)
from outlines_cascade.response import StructuredResponse

# ── test fixtures ───────────────────────────────────────────────────────


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


class MockAdapter(OutlinesModelAdapter):
    """Mock adapter that returns a canned response or raises an error."""

    def __init__(
        self,
        provider: str,
        model: str,
        result_text: str = '{"label": "positive", "confidence": 0.95}',
        raises: Exception | None = None,
        **kwargs,
    ):
        super().__init__(provider, model, **kwargs)
        self._result_text = result_text
        self._raises = raises

    def _build_model(self):
        return MagicMock()

    def _call_sync(self, prompt, output_type):
        if self._raises:
            raise self._raises
        return AdapterResult(text=self._result_text)


@pytest.fixture
def no_db_cascade():
    """Create a cascade with no DB (no cooldown tracking)."""

    def _make(entries, adapters_dict):
        cascade = StructuredCascade(entries=entries)
        # Inject mock adapters directly
        cascade._adapters = adapters_dict
        return cascade

    return _make


# ── success cases ───────────────────────────────────────────────────────


class TestCascadeSuccess:
    @pytest.mark.asyncio
    async def test_first_entry_succeeds(self, no_db_cascade):
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": MockAdapter("openai", "gpt-4o"),
        }
        cascade = no_db_cascade(entries, adapters)

        result = await cascade.generate("test prompt", Sentiment)

        assert isinstance(result, StructuredResponse)
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.output_type == "json"
        assert len(result.attempts) == 1
        assert result.attempts[0].status == "success"
        assert isinstance(result.value, Sentiment)
        assert result.value.label == "positive"

    @pytest.mark.asyncio
    async def test_failover_to_second_entry(self, no_db_cascade):
        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-3"),
        ]
        adapters = {
            "openai/gpt-4o": MockAdapter(
                "openai", "gpt-4o", raises=Exception("rate limited")
            ),
            "anthropic/claude-3": MockAdapter("anthropic", "claude-3"),
        }
        cascade = no_db_cascade(entries, adapters)

        result = await cascade.generate("test prompt", Sentiment)

        assert result.provider == "anthropic"
        assert result.model == "claude-3"
        assert len(result.attempts) == 2
        assert result.attempts[0].status == "failed"
        assert result.attempts[0].error == "rate limited"
        assert result.attempts[1].status == "success"

    @pytest.mark.asyncio
    async def test_raw_text_no_output_type(self, no_db_cascade):
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": MockAdapter(
                "openai", "gpt-4o", result_text="Hello world"
            ),
        }
        cascade = no_db_cascade(entries, adapters)

        result = await cascade.generate("test prompt", output_type=None)

        assert result.value == "Hello world"
        assert result.output_type == "json"  # default category


# ── type routing ────────────────────────────────────────────────────────


class TestTypeRouting:
    @pytest.mark.asyncio
    async def test_regex_skipped_on_cloud_only(self, no_db_cascade):
        """Cloud entries should be skipped for regex types."""
        from outlines.types import regex

        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(
                provider="local",
                model="phi-3",
                provider_kind=None,
            ),
        ]
        # Only the local adapter exists; the cloud adapter would fail anyway
        # because it's skipped
        adapters = {
            "local/phi-3": MockAdapter(
                "local",
                "phi-3",
                result_text="1234",
            ),
        }
        cascade = no_db_cascade(entries, adapters)

        # Mark local entry as steerable via provider config
        from outlines_cascade.config import ProviderConfig

        cascade._providers = {
            "local": ProviderConfig(type="transformers"),
        }

        result = await cascade.generate(
            "test", regex(r"\d{4}")
        )

        assert result.provider == "local"
        assert result.output_type == "regex"
        assert result.value == "1234"
        # First entry (cloud) should be skipped
        assert result.attempts[0].status == "skipped_type"

    @pytest.mark.asyncio
    async def test_all_skipped_raises_type_error(self, no_db_cascade):
        """If all entries are type-incompatible, raise TypeCompatibilityError."""
        from outlines.types import regex

        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-3"),
        ]
        adapters = {}
        cascade = no_db_cascade(entries, adapters)

        with pytest.raises(TypeCompatibilityError):
            await cascade.generate("test", regex(r"\d{4}"))


# ── failure cases ───────────────────────────────────────────────────────


class TestCascadeFailures:
    @pytest.mark.asyncio
    async def test_all_entries_fail(self, no_db_cascade):
        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-3"),
        ]
        adapters = {
            "openai/gpt-4o": MockAdapter(
                "openai", "gpt-4o", raises=Exception("timeout")
            ),
            "anthropic/claude-3": MockAdapter(
                "anthropic", "claude-3", raises=Exception("500 error")
            ),
        }
        cascade = no_db_cascade(entries, adapters)

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await cascade.generate("test", Sentiment)

        assert "timeout" in str(exc_info.value)
        assert "500 error" in str(exc_info.value)
        assert len(exc_info.value.attempts) == 2
        assert exc_info.value.attempts[0].status == "failed"
        assert exc_info.value.attempts[1].status == "failed"

    @pytest.mark.asyncio
    async def test_empty_entries_raises(self):
        cascade = StructuredCascade(entries=[])
        with pytest.raises(AllProvidersExhaustedError):
            await cascade.generate("test", Sentiment)

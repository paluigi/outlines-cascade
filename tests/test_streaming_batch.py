"""Tests for streaming and batch functionality."""

from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from outlines_cascade.adapters import AdapterResult, OutlinesModelAdapter
from outlines_cascade.config import CascadeEntry, ProviderConfig
from outlines_cascade.engine import StructuredCascade
from outlines_cascade.errors import AllProvidersExhaustedError
from outlines_cascade.response import (
    BatchResult,
    StreamChunk,
    StructuredResponse,
)

# ── test models ─────────────────────────────────────────────────────────


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


class StreamingMockAdapter(OutlinesModelAdapter):
    """Mock adapter that simulates streaming by splitting text into chunks."""

    def __init__(
        self,
        provider: str,
        model: str,
        full_text: str = '{"label": "positive", "confidence": 0.95}',
        chunk_size: int = 10,
        raises_on_stream: Exception | None = None,
        raises_on_generate: Exception | None = None,
        **kwargs,
    ):
        super().__init__(provider, model, **kwargs)
        self._full_text = full_text
        self._chunk_size = chunk_size
        self._raises_on_stream = raises_on_stream
        self._raises_on_generate = raises_on_generate

    def _build_model(self):
        return MagicMock()

    def _call_sync(self, prompt, output_type):
        if self._raises_on_generate:
            raise self._raises_on_generate
        return AdapterResult(text=self._full_text)

    def _stream_sync(self, prompt, output_type):
        if self._raises_on_stream:
            raise self._raises_on_stream
        text = self._full_text
        for i in range(0, len(text), self._chunk_size):
            yield text[i : i + self._chunk_size]

    def _batch_sync(self, prompts, output_type):
        return [AdapterResult(text=self._full_text) for _ in prompts]


# ── helpers ─────────────────────────────────────────────────────────────


def make_cascade(entries, adapters_dict, providers=None):
    """Create a cascade with pre-injected adapters."""
    cascade = StructuredCascade(entries=entries, providers=providers or {})
    cascade._adapters = adapters_dict
    return cascade


# ── streaming tests ─────────────────────────────────────────────────────


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming yields chunks then a final response."""
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                full_text='{"label": "positive", "confidence": 0.95}',
                chunk_size=20,
            ),
        }
        cascade = make_cascade(entries, adapters)

        items = []
        async for item in cascade.stream("test", Sentiment):
            items.append(item)

        # Last item should be StructuredResponse
        assert isinstance(items[-1], StructuredResponse)
        response = items[-1]
        assert response.provider == "openai"
        assert response.value.label == "positive"

        # Items before last should be StreamChunks
        chunks = [i for i in items if isinstance(i, StreamChunk)]
        assert len(chunks) >= 1

        # Reconstructed text should match
        full_text = "".join(c.text for c in chunks)
        assert "positive" in full_text

        # Second-to-last should be done=True sentinel
        assert chunks[-1].done is True

    @pytest.mark.asyncio
    async def test_stream_failover(self):
        """Test that streaming falls back to the next entry on failure."""
        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-3"),
        ]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                raises_on_stream=Exception("stream error"),
            ),
            "anthropic/claude-3": StreamingMockAdapter(
                "anthropic", "claude-3",
                full_text='{"label": "negative", "confidence": 0.8}',
                chunk_size=15,
            ),
        }
        cascade = make_cascade(entries, adapters)

        items = []
        async for item in cascade.stream("test", Sentiment):
            items.append(item)

        response = items[-1]
        assert isinstance(response, StructuredResponse)
        assert response.provider == "anthropic"

        # First attempt should have failed
        assert response.attempts[0].status == "failed"
        assert response.attempts[0].error == "stream error"

    @pytest.mark.asyncio
    async def test_stream_all_fail(self):
        """Test that streaming raises when all entries fail."""
        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
        ]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                raises_on_stream=Exception("connection refused"),
            ),
        }
        cascade = make_cascade(entries, adapters)

        with pytest.raises(AllProvidersExhaustedError):
            async for _ in cascade.stream("test", Sentiment):
                pass

    @pytest.mark.asyncio
    async def test_stream_raw_text(self):
        """Test streaming without an output_type (raw text)."""
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                full_text="Hello world this is a test",
                chunk_size=5,
            ),
        }
        cascade = make_cascade(entries, adapters)

        items = []
        async for item in cascade.stream("test", output_type=None):
            items.append(item)

        response = items[-1]
        assert isinstance(response, StructuredResponse)
        assert "Hello world" in response.value

    @pytest.mark.asyncio
    async def test_stream_skips_incompatible_type(self):
        """Test that streaming skips cloud entries for regex types."""
        from outlines.types import regex

        entries = [
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="local", model="phi-3"),
        ]
        adapters = {
            "local/phi-3": StreamingMockAdapter(
                "local", "phi-3",
                full_text="1234",
            ),
        }
        cascade = make_cascade(entries, adapters)
        cascade._providers = {
            "local": ProviderConfig(type="transformers"),
        }

        items = []
        async for item in cascade.stream("test", regex(r"\d{4}")):
            items.append(item)

        response = items[-1]
        assert response.provider == "local"
        assert response.output_type == "regex"
        # First entry should be skipped (type incompatible)
        assert response.attempts[0].status == "skipped_type"


# ── batch tests ─────────────────────────────────────────────────────────


class TestBatch:
    @pytest.mark.asyncio
    async def test_batch_basic(self):
        """Test batch processes multiple prompts."""
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                full_text='{"label": "positive", "confidence": 0.9}',
            ),
        }
        cascade = make_cascade(entries, adapters)

        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        results = await cascade.batch(prompts, Sentiment)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, BatchResult)
            assert result.response is not None
            assert result.error is None
            assert result.response.value.label == "positive"

    @pytest.mark.asyncio
    async def test_batch_partial_failure(self):
        """Test batch handles individual prompt failures gracefully."""
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                raises_on_generate=Exception("rate limited"),
            ),
        }
        cascade = make_cascade(entries, adapters)

        prompts = ["prompt 1", "prompt 2"]
        results = await cascade.batch(prompts, Sentiment)

        assert len(results) == 2
        for result in results:
            assert result.response is None
            assert result.error is not None
            assert "rate limited" in result.error

    @pytest.mark.asyncio
    async def test_batch_shares_adapter_cache(self):
        """Test that batch reuses the same adapter instance across prompts."""
        entries = [CascadeEntry(provider="openai", model="gpt-4o")]
        adapters = {
            "openai/gpt-4o": StreamingMockAdapter(
                "openai", "gpt-4o",
                full_text='{"label": "neutral", "confidence": 0.5}',
            ),
        }
        cascade = make_cascade(entries, adapters)

        prompts = ["p1", "p2"]
        await cascade.batch(prompts, Sentiment)

        # The adapter should still be the same instance (cached)
        assert "openai/gpt-4o" in cascade._adapters
        cached = cascade._adapters["openai/gpt-4o"]
        assert cached is adapters["openai/gpt-4o"]

    @pytest.mark.asyncio
    async def test_batch_empty(self):
        """Test batch with empty prompt list."""
        cascade = make_cascade(
            [CascadeEntry(provider="openai", model="gpt-4o")], {}
        )
        results = await cascade.batch([], Sentiment)
        assert results == []

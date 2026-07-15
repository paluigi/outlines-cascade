"""Integration tests for outlines-cascade.

These tests require real API keys and network access.  They are skipped
unless the relevant environment variables are set.

Run with:
    OPENAI_API_KEY=*** ANTHROPIC_API_KEY=*** pytest -m integration -v
"""

import os
from typing import Literal

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.integration


# ── shared models ───────────────────────────────────────────────────────


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def openai_available():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def anthropic_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


# ── OpenAI ──────────────────────────────────────────────────────────────


class TestOpenAIIntegration:
    @pytest.mark.asyncio
    async def test_pydantic_via_openai(self, openai_available):
        from outlines_cascade import (
            CascadeEntry,
            ProviderConfig,
            generate,
        )

        result = await generate(
            prompt="I absolutely love this product!",
            output_type=Sentiment,
            entries=[
                CascadeEntry(provider="openai", model="gpt-4o-mini")
            ],
            providers={
                "openai": ProviderConfig(
                    type="openai", api_key_env="OPENAI_API_KEY"
                ),
            },
        )
        assert result.provider == "openai"
        assert result.value.label in ("positive", "negative", "neutral")
        assert 0.0 <= result.value.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_choice_via_openai(self, openai_available):
        from outlines_cascade import (
            CascadeEntry,
            ProviderConfig,
            generate,
        )

        result = await generate(
            prompt="The weather is sunny today.",
            output_type=Literal["positive", "negative", "neutral"],
            entries=[
                CascadeEntry(provider="openai", model="gpt-4o-mini")
            ],
            providers={
                "openai": ProviderConfig(
                    type="openai", api_key_env="OPENAI_API_KEY"
                ),
            },
        )
        assert result.value in ("positive", "negative", "neutral")


# ── Anthropic ───────────────────────────────────────────────────────────


class TestAnthropicIntegration:
    @pytest.mark.asyncio
    async def test_pydantic_via_anthropic(self, anthropic_available):
        from outlines_cascade import (
            CascadeEntry,
            ProviderConfig,
            generate,
        )

        result = await generate(
            prompt="I love this product!",
            output_type=Sentiment,
            entries=[
                CascadeEntry(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                )
            ],
            providers={
                "anthropic": ProviderConfig(
                    type="anthropic", api_key_env="ANTHROPIC_API_KEY"
                ),
            },
        )
        assert result.provider == "anthropic"
        assert result.value.label in ("positive", "negative", "neutral")


# ── Cascade failover ────────────────────────────────────────────────────


class TestCascadeFailover:
    @pytest.mark.asyncio
    async def test_failover_on_bad_model(
        self, openai_available, anthropic_available
    ):
        from outlines_cascade import (
            CascadeEntry,
            ProviderConfig,
            generate,
        )

        result = await generate(
            prompt="I love this product!",
            output_type=Sentiment,
            entries=[
                CascadeEntry(
                    provider="openai", model="nonexistent-model-xxx"
                ),
                CascadeEntry(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                ),
            ],
            providers={
                "openai": ProviderConfig(
                    type="openai", api_key_env="OPENAI_API_KEY"
                ),
                "anthropic": ProviderConfig(
                    type="anthropic", api_key_env="ANTHROPIC_API_KEY"
                ),
            },
        )
        assert result.provider == "anthropic"
        assert len(result.attempts) == 2
        assert result.attempts[0].status == "failed"
        assert result.attempts[1].status == "success"


# ── Streaming ───────────────────────────────────────────────────────────


class TestStreamingIntegration:
    @pytest.mark.asyncio
    async def test_stream_via_openai(self, openai_available):
        from outlines_cascade import (
            CascadeEntry,
            ProviderConfig,
            StreamChunk,
            StructuredResponse,
            stream,
        )

        items = []
        async for item in stream(
            prompt="I love this product!",
            output_type=Sentiment,
            entries=[
                CascadeEntry(provider="openai", model="gpt-4o-mini")
            ],
            providers={
                "openai": ProviderConfig(
                    type="openai", api_key_env="OPENAI_API_KEY"
                ),
            },
        ):
            items.append(item)

        chunks = [i for i in items if isinstance(i, StreamChunk)]
        response = items[-1]
        assert isinstance(response, StructuredResponse)
        assert response.value.label in ("positive", "negative", "neutral")
        assert len(chunks) >= 1

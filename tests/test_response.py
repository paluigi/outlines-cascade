"""Tests for the StructuredResponse and CascadeAttempt models."""

from datetime import datetime

from outlines_cascade.response import CascadeAttempt, StructuredResponse


class TestCascadeAttempt:
    def test_basic_creation(self):
        a = CascadeAttempt(
            provider="openai",
            model="gpt-4o",
            status="success",
            latency_ms=150,
        )
        assert a.provider == "openai"
        assert a.model == "gpt-4o"
        assert a.status == "success"
        assert a.latency_ms == 150
        assert a.error is None

    def test_entry_key(self):
        a = CascadeAttempt(
            provider="anthropic", model="claude-3", status="failed"
        )
        assert a.entry_key == "anthropic/claude-3"

    def test_with_error(self):
        a = CascadeAttempt(
            provider="openai",
            model="gpt-4o",
            status="failed",
            error="HTTP 429",
        )
        assert a.error == "HTTP 429"


class TestStructuredResponse:
    def test_basic_creation(self):
        r = StructuredResponse(
            value="positive",
            provider="openai",
            model="gpt-4o",
            output_type="regex",
        )
        assert r.value == "positive"
        assert r.provider == "openai"
        assert r.model == "gpt-4o"
        assert r.output_type == "regex"
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.latency_ms == 0
        assert isinstance(r.timestamp, datetime)
        assert r.attempts == []

    def test_entry_key(self):
        r = StructuredResponse(
            value=42,
            provider="ollama",
            model="llama3.1",
            output_type="json",
        )
        assert r.entry_key == "ollama/llama3.1"

    def test_with_attempts(self):
        attempts = [
            CascadeAttempt(provider="openai", model="gpt-4o", status="failed"),
            CascadeAttempt(
                provider="anthropic", model="claude-3", status="success"
            ),
        ]
        r = StructuredResponse(
            value="ok",
            provider="anthropic",
            model="claude-3",
            output_type="choice",
            attempts=attempts,
        )
        assert len(r.attempts) == 2
        assert r.succeeded_entry is not None
        assert r.succeeded_entry.provider == "anthropic"

    def test_succeeded_entry_none_when_no_success(self):
        attempts = [
            CascadeAttempt(provider="openai", model="gpt-4o", status="failed"),
        ]
        r = StructuredResponse(
            value="x",
            provider="openai",
            model="gpt-4o",
            output_type="json",
            attempts=attempts,
        )
        assert r.succeeded_entry is None

    def test_with_pydantic_value(self):
        from typing import Literal

        from pydantic import BaseModel

        class Result(BaseModel):
            label: Literal["positive", "negative"]

        val = Result(label="positive")
        r = StructuredResponse[Result](
            value=val,
            provider="openai",
            model="gpt-4o",
            output_type="json",
            input_tokens=10,
            output_tokens=5,
        )
        assert r.value.label == "positive"
        assert r.input_tokens == 10

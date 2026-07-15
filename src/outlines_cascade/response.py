"""Result objects for structured cascade generation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class CascadeAttempt(BaseModel):
    """Record of a single provider attempt within the cascade.

    Attributes:
        provider: Name of the provider (e.g. "openai").
        model: Model identifier (e.g. "gpt-4o").
        status: Outcome — "success", "skipped_type", "skipped_cooldown",
            or "failed".
        latency_ms: Wall-clock time for this attempt in milliseconds.
        error: Error message if status is "failed".
    """

    provider: str
    model: str
    status: str
    latency_ms: int = 0
    error: str | None = None

    @property
    def entry_key(self) -> str:
        """Return the ``provider/model`` identifier."""
        return f"{self.provider}/{self.model}"


class StructuredResponse(BaseModel, Generic[T]):
    """The result of a structured cascade generation.

    Encapsulates the parsed/validated structured output together with
    metadata about which provider produced it and the full attempt history.

    Attributes:
        value: The parsed structured output (Pydantic instance, validated
            string for regex, etc.).
        provider: Name of the provider that succeeded.
        model: Model identifier that generated the response.
        input_tokens: Token count of the prompt.
        output_tokens: Token count of the completion.
        output_type: The Outlines term type used (e.g. "JsonSchema",
            "Regex", "CFG", "Choice").
        attempts: Ordered list of all cascade attempts (for observability).
        latency_ms: Total wall-clock time for the entire cascade.
        timestamp: When the result was generated (UTC).
    """

    value: Any
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    output_type: str
    attempts: list[CascadeAttempt] = Field(default_factory=list)
    latency_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def entry_key(self) -> str:
        """Return the ``provider/model`` identifier of the winning entry."""
        return f"{self.provider}/{self.model}"

    @property
    def succeeded_entry(self) -> CascadeAttempt | None:
        """Return the attempt that succeeded, or None."""
        for a in self.attempts:
            if a.status == "success":
                return a
        return None


class StreamChunk(BaseModel):
    """A single chunk yielded during streaming generation.

    Attributes:
        text: The text content of this chunk.
        provider: Provider that generated this chunk (None until a
            provider succeeds and starts streaming).
        model: Model that generated this chunk (None until streaming
            starts).
        done: True if this is the last chunk before the final
            StructuredResponse.
    """

    text: str
    provider: str | None = None
    model: str | None = None
    done: bool = False


class BatchResult(BaseModel, Generic[T]):
    """Result of a single prompt within a batch cascade run.

    Attributes:
        response: The StructuredResponse if successful, or None if this
            prompt failed.
        error: Error message if the cascade failed for this prompt.
        prompt: The original prompt.
    """

    response: StructuredResponse[T] | None = None
    error: str | None = None
    prompt: str

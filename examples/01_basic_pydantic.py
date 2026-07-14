"""Example: basic Pydantic model structured generation.

Run with:
    python examples/01_basic_pydantic.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from outlines_cascade import CascadeEntry, generate


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def main():
    result = await generate(
        prompt="I absolutely love this product! Best purchase ever.",
        output_type=Sentiment,
        entries=[
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
        ],
    )
    print(f"Value:     {result.value}")
    print(f"Provider:  {result.provider}")
    print(f"Model:     {result.model}")
    print(f"Tokens:    {result.input_tokens} in / {result.output_tokens} out")
    print(f"Latency:   {result.latency_ms}ms")
    print(f"Attempts:  {len(result.attempts)}")
    for a in result.attempts:
        print(f"  - {a.entry_key}: {a.status} ({a.latency_ms}ms)")


if __name__ == "__main__":
    asyncio.run(main())

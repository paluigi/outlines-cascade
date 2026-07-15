"""Example: streaming structured generation with cascade failover.

Run with:
    python examples/02_streaming.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from outlines_cascade import CascadeEntry, StreamChunk, StructuredResponse, stream


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str


async def main():
    print("Starting stream...\n")

    async for item in stream(
        prompt="The new economic policy has significantly reduced unemployment.",
        output_type=Sentiment,
        entries=[
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
        ],
    ):
        if isinstance(item, StreamChunk):
            if not item.done:
                print(item.text, end="", flush=True)
        elif isinstance(item, StructuredResponse):
            print("\n")
            print("--- Final Response ---")
            print(f"Value:     {item.value}")
            print(f"Provider:  {item.provider}")
            print(f"Model:     {item.model}")
            print(f"Latency:   {item.latency_ms}ms")
            print(f"Attempts:  {len(item.attempts)}")
            for a in item.attempts:
                print(f"  - {a.entry_key}: {a.status}")


if __name__ == "__main__":
    asyncio.run(main())

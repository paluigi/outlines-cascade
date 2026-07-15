"""Example: mixed cascade with cloud + local fallback and production DB.

Shows the full feature set:
- TOML config-driven setup
- Cloud entries for JSON Schema (auto-detected)
- Local entries for regex/CFG (auto-detected)
- SQLite cooldown tracking with exponential backoff
- Failed-prompt persistence

Requires: pip install "outlines-cascade[openai,anthropic,transformers,db]"
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from outlines_cascade import generate, load_config


class Priority(BaseModel):
    level: Literal["low", "medium", "high", "urgent"]
    category: str
    requires_manager: bool


async def main():
    # Load cascade config from TOML file
    # (see config.example.toml for the format)
    config = load_config("~/.config/outlines-cascade/config.toml")

    result = await generate(
        prompt=(
            "Customer email: 'My account was charged twice and I can't "
            "log in! This is the third time I've contacted support!'"
        ),
        output_type=Priority,
        config=config,
        cascade_name="primary",  # looked up from TOML
    )

    print(f"Value:     {result.value}")
    print(f"Provider:  {result.provider}")
    print(f"Model:     {result.model}")
    print(f"Tokens:    {result.input_tokens} in / {result.output_tokens} out")
    print(f"Latency:   {result.latency_ms}ms")
    print(f"Attempts:  {len(result.attempts)}")
    for a in result.attempts:
        status = a.status
        if a.error:
            status += f" ({a.error})"
        print(f"  - {a.entry_key}: {status} [{a.latency_ms}ms]")


if __name__ == "__main__":
    asyncio.run(main())

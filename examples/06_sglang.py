"""Example: SGLang server for regex/CFG structured generation.

SGLang is a server-based provider that — unlike cloud APIs — supports
ALL output types (JSON Schema, regex, CFG) via Outlines' backends.

Requires: pip install outlines-cascade[sglang]
          + a running SGLang server (e.g. http://localhost:30000)
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from outlines_cascade import CascadeEntry, ProviderConfig, generate


class CustomerIssue(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str


async def main():
    result = await generate(
        prompt=(
            "Customer: 'I was double-charged on my last invoice "
            "and need a refund immediately!'"
        ),
        output_type=CustomerIssue,
        entries=[
            # SGLang supports all output types — no type skipping
            CascadeEntry(
                provider="sglang-server",
                model="meta-llama/Llama-3.1-8B-Instruct",
            ),
        ],
        providers={
            "sglang-server": ProviderConfig(
                type="sglang",
                base_url="http://localhost:30000/v1",
            ),
        },
    )

    print(f"Value:     {result.value}")
    print(f"Provider:  {result.provider}")
    print(f"Model:     {result.model}")


if __name__ == "__main__":
    asyncio.run(main())

"""Example: regex-constrained generation with local models.

Cloud APIs (OpenAI, Anthropic, Gemini) cannot enforce regex patterns.
This example shows how outlines-cascade automatically routes regex
requests to steerable (local) models, skipping incompatible cloud entries.

Requires: pip install outlines-cascade[transformers]
"""

import asyncio

from outlines.types import regex

from outlines_cascade import CascadeEntry, ProviderConfig, generate


async def main():
    # Define a phone-number regex pattern
    phone_pattern = regex(r"[0-9]{3}-[0-9]{3}-[0-9]{4}")

    result = await generate(
        prompt="Generate a phone number for the contact form.",
        output_type=phone_pattern,
        entries=[
            # This entry will be SKIPPED — cloud APIs can't enforce regex
            CascadeEntry(provider="openai", model="gpt-4o"),

            # This entry will be used — local models enforce regex via FSM
            CascadeEntry(
                provider="local",
                model="microsoft/Phi-3-mini-4k-instruct",
            ),
        ],
        providers={
            "openai": ProviderConfig(type="openai", api_key_env="OPENAI_API_KEY"),
            "local": ProviderConfig(type="transformers"),
        },
    )

    print(f"Value:     {result.value}")
    print(f"Provider:  {result.provider}")
    print(f"Model:     {result.model}")
    print(f"Output type: {result.output_type}")

    # The first entry should be skipped due to type incompatibility
    for a in result.attempts:
        print(f"  {a.entry_key}: {a.status}")


if __name__ == "__main__":
    asyncio.run(main())

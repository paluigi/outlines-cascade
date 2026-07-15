"""Example: batch processing of multiple prompts.

Shows how to classify multiple texts through the cascade in one call,
with graceful per-prompt failure handling.
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from outlines_cascade import CascadeEntry, batch


class Classification(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    category: Literal["product", "service", "shipping", "other"]


async def main():
    reviews = [
        "The product arrived on time and works perfectly!",
        "Customer service was rude and unhelpful.",
        "Shipping took 3 weeks, but the item is decent.",
        "Amazing quality, exceeded my expectations!",
        "The packaging was damaged but the product is fine.",
    ]

    results = await batch(
        prompts=reviews,
        output_type=Classification,
        entries=[
            CascadeEntry(provider="openai", model="gpt-4o-mini"),
        ],
    )

    for review, result in zip(reviews, results, strict=True):
        if result.response:
            v = result.response.value
            print(f"[{v.sentiment:>8}] [{v.category:>8}] {review}")
        else:
            print(f"[  ERROR] {review}: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())

# Quick Start

## Installation

```bash
pip install outlines-cascade
```

With provider-specific dependencies:

```bash
pip install "outlines-cascade[openai,anthropic,db]"
```

## First Generation

```python
import asyncio
from typing import Literal
from pydantic import BaseModel
from outlines_cascade import generate, CascadeEntry

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

async def main():
    result = await generate(
        prompt="I absolutely love this product!",
        output_type=Sentiment,
        entries=[
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
        ],
    )
    print(result.value)       # Sentiment(label="positive", confidence=0.95)
    print(result.provider)    # "openai"

asyncio.run(main())
```

## Streaming

```python
from outlines_cascade import stream, StreamChunk, StructuredResponse

async for item in stream(prompt="...", output_type=Sentiment, entries=[...]):
    if isinstance(item, StreamChunk) and not item.done:
        print(item.text, end="", flush=True)
    elif isinstance(item, StructuredResponse):
        print(f"\nProvider: {item.provider}")
```

## Batch

```python
from outlines_cascade import batch

results = await batch(
    prompts=["review 1", "review 2"],
    output_type=Sentiment,
    entries=[...],
)
```

## Config-Driven

```python
from outlines_cascade import generate, load_config

config = load_config("~/.config/outlines-cascade/config.toml")
result = await generate(
    prompt="...",
    output_type=Sentiment,
    config=config,
    cascade_name="primary",
)
```

See [Configuration](configuration.md) for the TOML format.

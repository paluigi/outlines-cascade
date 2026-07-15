# outlines-cascade

**Structured LLM generation with cascading failover.**

Combines [Outlines](https://github.com/dottxt-ai/outlines)' structured generation with [llm-pycascade](https://github.com/paluigi/llm-pycascade)'s cascading failover to enforce structured output (Pydantic, JSON Schema, regex, CFG, choice) across a fallback chain of LLM providers, returning a metadata-enriched result object.

> **Status:** Alpha — under active development.

## Installation

```bash
pip install outlines-cascade
```

With optional provider support:

```bash
pip install "outlines-cascade[openai,anthropic]"
```

## Quick Start

### Single generation

```python
import asyncio
from pydantic import BaseModel
from typing import Literal
from outlines_cascade import generate, CascadeEntry

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

async def main():
    result = await generate(
        prompt="I love this product!",
        output_type=Sentiment,
        entries=[
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
        ],
    )
    print(result.value)       # Sentiment(label="positive", confidence=0.95)
    print(result.provider)    # "openai"
    print(result.attempts)    # full attempt history

asyncio.run(main())
```

### Streaming

```python
from outlines_cascade import stream, StreamChunk, StructuredResponse

async for item in stream(
    prompt="Explain quantum computing.",
    entries=[CascadeEntry(provider="openai", model="gpt-4o")],
):
    if isinstance(item, StreamChunk) and not item.done:
        print(item.text, end="", flush=True)
    elif isinstance(item, StructuredResponse):
        print(f"\nProvider: {item.provider}, Latency: {item.latency_ms}ms")
```

### Batch

```python
from outlines_cascade import batch

results = await batch(
    prompts=["Review 1...", "Review 2...", "Review 3..."],
    output_type=Sentiment,
    entries=[CascadeEntry(provider="openai", model="gpt-4o")],
)
for r in results:
    if r.response:
        print(r.response.value)
    else:
        print(f"Failed: {r.error}")
```

## License

MIT

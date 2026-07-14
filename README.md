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

## License

MIT

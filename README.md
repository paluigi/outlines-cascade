# outlines-cascade

**Structured LLM generation with cascading failover.**

Combine [Outlines](https://github.com/dottxt-ai/outlines)' structured generation with [llm-pycascade](https://github.com/paluigi/llm-pycascade)'s cascading failover — enforce structured output (Pydantic, JSON Schema, regex, CFG, choice) across a fallback chain of LLM providers, with a metadata-enriched result object.

## Features

- **Full structured output** — Pydantic models, JSON Schema, `Literal`/choice, regex, context-free grammars
- **Cascade failover** — ordered provider chain; if one fails, the next is tried immediately
- **Circuit breaking** — exponential backoff cooldowns (30s → 3600s) with `Retry-After` support
- **Smart type routing** — regex/CFG requests auto-skip cloud APIs and route to local models
- **Streaming** — incremental text chunks via async generators
- **Batch** — process multiple prompts with shared adapter cache
- **7 providers** — OpenAI, Anthropic, Gemini, Ollama, SGLang, Transformers, LlamaCpp
- **Async-native** — built on `asyncio` throughout

## Installation

```bash
pip install outlines-cascade
```

With provider support:

```bash
pip install "outlines-cascade[openai,anthropic,db]"
```

## Quick Start

### Single generation

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
        prompt="I love this product!",
        output_type=Sentiment,
        entries=[
            CascadeEntry(provider="openai", model="gpt-4o"),
            CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
        ],
    )
    print(result.value)       # Sentiment(label="positive", confidence=0.95)
    print(result.provider)    # "openai"
    print(result.attempts)    # [CascadeAttempt(...)]

asyncio.run(main())
```

### Streaming

```python
from outlines_cascade import stream, StreamChunk, StructuredResponse

async for item in stream(prompt="...", output_type=Sentiment, entries=[...]):
    if isinstance(item, StreamChunk) and not item.done:
        print(item.text, end="", flush=True)
    elif isinstance(item, StructuredResponse):
        print(f"\nProvider: {item.provider}")
```

### Batch

```python
from outlines_cascade import batch

results = await batch(
    prompts=["review 1", "review 2"],
    output_type=Sentiment,
    entries=[...],
)
for r in results:
    print(r.response.value if r.response else r.error)
```

### Config-driven (TOML)

```python
from outlines_cascade import generate, load_config

config = load_config("~/.config/outlines-cascade/config.toml")
result = await generate(prompt="...", output_type=Sentiment, config=config, cascade_name="primary")
```

See [`config.example.toml`](config.example.toml) for the full format.

## Type Routing

Cloud APIs (OpenAI, Anthropic, Gemini) only support JSON Schema. Local models and SGLang support **all** types via FSM-based constrained decoding:

| Type | Cloud | Local / SGLang |
|------|:-----:|:--------------:|
| Pydantic / JSON Schema | ✅ | ✅ |
| `Literal` / choice | ✅ | ✅ |
| Regex | ❌ skipped | ✅ |
| CFG | ❌ skipped | ✅ |

When you request a regex with a mixed cascade, cloud entries are **silently skipped** and the request routes to the local model automatically.

## Providers

| Provider | Type | Supports |
|----------|------|----------|
| OpenAI | cloud | JSON Schema, Choice |
| Anthropic | cloud | JSON Schema, Choice |
| Gemini | cloud | JSON Schema, Choice |
| Ollama | cloud | JSON Schema, Choice |
| SGLang | steerable | All (JSON, regex, CFG) |
| Transformers | steerable | All (FSM) |
| LlamaCpp | steerable | All (FSM) |

## Documentation

- [Quick Start](docs/quickstart.md)
- [User Guide](docs/user_guide.md) — type routing, cascade patterns, cooldowns
- [Configuration](docs/configuration.md) — TOML format reference
- [API Reference](docs/api_reference.md) — all public classes and functions
- [Provider Reference](docs/providers.md) — provider setup and capabilities
- [Examples](examples/) — runnable demos

## Dependencies

- [Outlines](https://github.com/dottxt-ai/outlines) `>=1.0` — structured generation
- [llm-pycascade](https://github.com/paluigi/llm-pycascade) `>=0.1.0` — cascade failover
- [Pydantic](https://github.com/pydantic/pydantic) `>=2.0` — data validation

## License

MIT

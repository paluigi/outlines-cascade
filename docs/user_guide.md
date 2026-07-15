# User Guide

## Core Concepts

### The Cascade

A **cascade** is an ordered list of provider/model pairs. The engine tries each entry in sequence until one succeeds:

1. **Type check** — skip entries that don't support the requested output type
2. **Cooldown check** — skip entries that recently failed (exponential backoff)
3. **Generate** — call the model via the Outlines adapter
4. **On failure** — set cooldown, move to next entry
5. **On success** — parse result, return `StructuredResponse`

### Output Types and Type Routing

outlines-cascade supports the full Outlines type surface:

| Type | Example | Cloud APIs | Local models |
|------|---------|------------|--------------|
| **Pydantic model** | `class Sentiment(BaseModel)` | ✅ JSON Schema | ✅ FSM |
| **JSON Schema dict** | `{"type": "object", ...}` | ✅ | ✅ |
| **Literal / choice** | `Literal["yes", "no"]` | ✅ (converted) | ✅ |
| **Regex** | `regex(r"\d{4}")` | ❌ skipped | ✅ FSM |
| **CFG** | `cfg("root := ...")` | ❌ skipped | ✅ FSM |

**How it works:**

- Cloud APIs (OpenAI, Anthropic, Gemini, Ollama remote) only support JSON Schema via `response_format`. When you request a regex or CFG, cloud entries are **silently skipped** and the cascade falls through to steerable (local) models.
- Steerable models (Transformers, LlamaCpp, SGLang) support **all types** via FSM-based constrained decoding (logits processor at generation time).
- `Literal`/choice types are automatically converted to a JSON-compatible form for cloud providers.

**Overriding auto-detection:**

You can explicitly set `supported_types` on a cascade entry:

```python
CascadeEntry(
    provider="local",
    model="phi-3",
    supported_types=["json", "regex"],  # restrict this entry
)
```

### The StructuredResponse

Every generation returns a `StructuredResponse`:

```python
result.value          # The parsed output (Pydantic instance, str, etc.)
result.provider       # "openai"
result.model          # "gpt-4o"
result.output_type    # "json", "regex", "cfg", "choice"
result.attempts       # [CascadeAttempt(...), ...]
result.latency_ms     # Total cascade wall-clock time
result.input_tokens   # (if available)
result.output_tokens  # (if available)
```

The `attempts` list records every entry tried, with status and latency — full observability of the cascade path.

## Patterns

### Cloud-First with Local Fallback

```python
entries = [
    CascadeEntry(provider="openai", model="gpt-4o"),
    CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
    CascadeEntry(provider="local", model="microsoft/Phi-3-mini-4k-instruct"),
]
```

For JSON Schema types, all three entries are tried in order. For regex/CFG, only the local model is tried (cloud entries are skipped automatically).

### Regex-Only (Local Only)

```python
from outlines.types import regex

result = await generate(
    prompt="Generate a ZIP code",
    output_type=regex(r"\d{5}"),
    entries=[
        CascadeEntry(provider="local", model="microsoft/Phi-3-mini-4k-instruct"),
    ],
    providers={"local": ProviderConfig(type="transformers")},
)
# result.value is guaranteed to match \d{5}
```

### SGLang as a Universal Backend

SGLang is special — it's a server-based provider that supports all output types:

```python
entries = [
    CascadeEntry(provider="sglang", model="meta-llama/Llama-3.1-8B-Instruct"),
]
providers = {
    "sglang": ProviderConfig(type="sglang", base_url="http://localhost:30000/v1"),
}
```

SGLang handles regex via `extra_body: {"regex": ...}` and CFG via `extra_body: {"ebnf": ...}`.

## Cooldowns and Backoff

When a provider fails, it's put on cooldown:

| Failure # | Cooldown |
|-----------|----------|
| 1st | 30s |
| 2nd | 60s |
| 3rd | 120s |
| 4th | 240s |
| 5th | 480s |
| 6th+ | 960s → capped at 3600s |

HTTP 429 responses with a `Retry-After` header use the header value directly.

Cooldowns require a SQLite database (`db_path` parameter). Without a DB, failover still works but there's no circuit breaking.

## Streaming

Streaming yields `StreamChunk` objects incrementally, then a final `StructuredResponse`:

```python
async for item in cascade.stream(prompt, output_type=MyModel):
    match item:
        case StreamChunk(done=False):
            print(item.text, end="", flush=True)
        case StreamChunk(done=True):
            pass  # stream complete
        case StructuredResponse():
            print(f"\nFinal value: {item.value}")
```

If the first provider's stream fails mid-way, the cascade falls back to the next entry, restarting the stream.

## Batch

Batch runs the full cascade for each prompt independently:

```python
results = await cascade.batch(["prompt 1", "prompt 2"], output_type=MyModel)
for r in results:
    if r.response:
        print(r.response.value)
    else:
        print(f"Failed: {r.error}")
```

All prompts share the adapter cache (no redundant model loading) and cooldown state.

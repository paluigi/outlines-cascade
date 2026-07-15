# API Reference

## Entry Points

### `generate(prompt, output_type, entries, ...)`

```python
async def generate(
    prompt: str,
    output_type: Any | None = None,
    entries: list[CascadeEntry] | None = None,
    providers: dict[str, ProviderConfig] | None = None,
    db_path: str | None = None,
    failure_dir: str | None = None,
    config: AppConfig | None = None,
    cascade_name: str | None = None,
    **inference_kwargs: Any,
) -> StructuredResponse
```

Generate a structured response via cascade failover.

**Returns:** `StructuredResponse` with parsed value and metadata.

**Raises:**
- `AllProvidersExhaustedError` — all entries failed or were skipped
- `TypeCompatibilityError` — no entry supports the requested output type
- `ConfigError` — missing entries or cascade name

---

### `stream(prompt, output_type, entries, ...)`

```python
async def stream(...) -> AsyncIterator[StreamChunk | StructuredResponse]
```

Stream a structured response. Yields `StreamChunk` objects, then a final `StructuredResponse` as the last item.

---

### `batch(prompts, output_type, entries, ...)`

```python
async def batch(...) -> list[BatchResult]
```

Generate structured responses for multiple prompts. Each prompt runs through the full cascade independently.

**Returns:** List of `BatchResult`, one per prompt, in order.

---

## Result Types

### `StructuredResponse[T]`

| Attribute | Type | Description |
|-----------|------|-------------|
| `value` | `T` | Parsed output (Pydantic instance, str, etc.) |
| `provider` | `str` | Provider that succeeded |
| `model` | `str` | Model identifier |
| `output_type` | `str` | Category: `"json"`, `"regex"`, `"cfg"`, `"choice"` |
| `attempts` | `list[CascadeAttempt]` | All entries tried |
| `latency_ms` | `int` | Total cascade wall-clock time |
| `input_tokens` | `int` | Prompt tokens (if available) |
| `output_tokens` | `int` | Completion tokens (if available) |
| `timestamp` | `datetime` | When generated (UTC) |

**Properties:**
- `entry_key` → `"provider/model"`
- `succeeded_entry` → the `CascadeAttempt` that succeeded

### `CascadeAttempt`

| Attribute | Type | Description |
|-----------|------|-------------|
| `provider` | `str` | Provider name |
| `model` | `str` | Model identifier |
| `status` | `str` | `"success"`, `"failed"`, `"skipped_type"`, `"skipped_cooldown"` |
| `latency_ms` | `int` | Wall-clock time for this attempt |
| `error` | `str | None` | Error message if failed |

### `StreamChunk`

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text chunk content |
| `provider` | `str | None` | Provider (set during streaming) |
| `model` | `str | None` | Model (set during streaming) |
| `done` | `bool` | `True` on the final sentinel chunk |

### `BatchResult[T]`

| Attribute | Type | Description |
|-----------|------|-------------|
| `response` | `StructuredResponse[T] | None` | Result if successful |
| `error` | `str | None` | Error message if failed |
| `prompt` | `str` | The original prompt |

---

## Config Types

### `CascadeEntry`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | required | Provider key |
| `model` | `str` | required | Model identifier |
| `supported_types` | `list[str] | None` | auto | Override type support |
| `provider_kind` | `ProviderKind | None` | auto | `"cloud"` or `"steerable"` |
| `device` | `str | None` | None | Device for Transformers |
| `base_url` | `str | None` | None | API base URL override |

### `ProviderConfig`

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | `str` | Provider type (openai, anthropic, etc.) |
| `api_key_env` | `str | None` | Env var name for API key |
| `api_key_service` | `str | None` | Keyring service name |
| `base_url` | `str | None` | API base URL override |

### `AppConfig`

| Attribute | Type | Description |
|-----------|------|-------------|
| `providers` | `dict[str, ProviderConfig]` | Provider configurations |
| `cascades` | `dict[str, CascadeConfig]` | Named cascade definitions |
| `database` | `DatabaseConfig` | SQLite DB config |
| `failure_persistence` | `FailureConfig` | Failed-prompt persistence |

### `load_config(path=None)`

Load configuration from a TOML file. If `path` is None, searches:
1. `OUTLINES_CASCADE_CONFIG` env var
2. `~/.config/outlines-cascade/config.toml`

---

## Engine

### `StructuredCascade`

```python
cascade = StructuredCascade(
    entries: list[CascadeEntry],
    providers: dict[str, ProviderConfig] = None,
    db_path: str = None,
    failure_dir: str = None,
)
```

**Methods:**
- `async generate(prompt, output_type=None) -> StructuredResponse`
- `async stream(prompt, output_type=None) -> AsyncIterator[StreamChunk | StructuredResponse]`
- `async batch(prompts, output_type=None) -> list[BatchResult]`

---

## Adapters

### `OutlinesModelAdapter`

Abstract base class for all model adapters. Each adapter wraps an Outlines model behind a unified async interface.

**Methods:**
- `async generate(prompt, output_type) -> AdapterResult`
- `async stream(prompt, output_type) -> AsyncIterator[str]`
- `async batch(prompts, output_type) -> list[AdapterResult]`

### `build_adapter(provider_type, provider_name, model, ...)`

Factory function to create the correct adapter for a provider type.

Supported types: `openai`, `anthropic`, `gemini`, `ollama`, `sglang`, `transformers`, `llamacpp`.

---

## Errors

| Class | When |
|-------|------|
| `OutlinesCascadeError` | Base for all errors |
| `AllProvidersExhaustedError` | All entries failed |
| `TypeCompatibilityError` | No entry supports the output type |
| `AdapterError` | Adapter build/call failure |
| `ConfigError` | Configuration error |

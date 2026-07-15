# Configuration

outlines-cascade can be configured via TOML file or programmatically.

## TOML Configuration

Default location: `~/.config/outlines-cascade/config.toml`
Override with: `OUTLINES_CASCADE_CONFIG=/path/to/config.toml`

```toml
# ── Providers ──

[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[providers.local]
type = "transformers"

[providers.sglang]
type = "sglang"
base_url = "http://localhost:30000/v1"

# ── Cascades ──

[cascades.primary]
entries = [
    { provider = "openai", model = "gpt-4o" },
    { provider = "anthropic", model = "claude-sonnet-4-20250514" },
    { provider = "local", model = "microsoft/Phi-3-mini-4k-instruct" },
]

[cascades.fast]
entries = [
    { provider = "openai", model = "gpt-4o-mini" },
    { provider = "local", model = "microsoft/Phi-3-mini-4k-instruct" },
]

# ── Database (optional, enables cooldowns) ──

[database]
path = "~/.local/share/outlines-cascade/db.sqlite"

# ── Failure persistence (optional) ──

[failure_persistence]
dir = "~/.local/share/outlines-cascade/failed_prompts"
```

## Cascade Entry Fields

Each entry in a cascade supports these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | required | Key matching a provider in `[providers]` |
| `model` | `str` | required | Model identifier |
| `supported_types` | `list[str]` | auto | Override type support: `"json"`, `"regex"`, `"cfg"`, `"choice"` |
| `provider_kind` | `str` | auto | `"cloud"` or `"steerable"` |
| `device` | `str` | `"auto"` | Device for Transformers (`"cuda"`, `"cpu"`) |
| `base_url` | `str` | None | Override provider API base URL |

## Programmatic Configuration

```python
from outlines_cascade import (
    AppConfig, CascadeConfig, CascadeEntry,
    ProviderConfig, DatabaseConfig, FailureConfig,
)

config = AppConfig(
    providers={
        "openai": ProviderConfig(type="openai", api_key_env="OPENAI_API_KEY"),
    },
    cascades={
        "primary": CascadeConfig(entries=[
            CascadeEntry(provider="openai", model="gpt-4o"),
        ]),
    },
    database=DatabaseConfig(path="/tmp/cascade.db"),
)
```

## Provider Types

| Type | Kind | API Key | Supports |
|------|------|---------|----------|
| `openai` | cloud | required (env) | JSON Schema, Choice |
| `anthropic` | cloud | required (env) | JSON Schema, Choice |
| `gemini` | cloud | required (env) | JSON Schema, Choice |
| `ollama` | cloud | none | JSON Schema, Choice |
| `sglang` | steerable | none (dummy) | All (JSON, regex, CFG, choice) |
| `transformers` | steerable | n/a | All (FSM) |
| `llamacpp` | steerable | n/a | All (FSM) |

## API Key Resolution

Keys are resolved in order:

1. Environment variable specified by `api_key_env`
2. Default env var: `{PROVIDER_NAME_UPPER}_API_KEY` (e.g. `OPENAI_API_KEY`)

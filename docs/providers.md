# Provider Reference

## Cloud Providers (Black-Box)

Cloud APIs receive the output type as a JSON Schema via `response_format`. They support JSON Schema and Choice (auto-converted), but **not** regex or CFG.

### OpenAI

```toml
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
```

- **Auth:** Bearer token via env var
- **Structured output:** `response_format: {json_schema: {strict: true}}`
- **Compatible:** Any OpenAI-compatible endpoint (vLLM, LiteLLM, Together AI) via `base_url`

**Install:** `pip install outlines-cascade[openai]`

### Anthropic

```toml
[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
```

- **Auth:** `x-api-key` header
- **Structured output:** Tool-based JSON extraction via Outlines adapter

**Install:** `pip install outlines-cascade[anthropic]`

### Google Gemini

```toml
[providers.gemini]
type = "gemini"
api_key_env = "GEMINI_API_KEY"
```

- **Auth:** API key as query parameter
- **Structured output:** `responseSchema` parameter

**Install:** `pip install outlines-cascade[gemini]`

### Ollama (Remote)

```toml
[providers.ollama]
type = "ollama"
# No API key needed
```

- **Auth:** None
- **Structured output:** `format: {json_schema}`

**Install:** `pip install outlines-cascade[ollama]`

---

## Steerable Providers

Steerable models enforce structure at the generation level (FSM-based logits processor) — they support **all** output types including regex and CFG.

### SGLang

```toml
[providers.sglang]
type = "sglang"
base_url = "http://localhost:30000/v1"
```

- **Auth:** Dummy key (`EMPTY` by default)
- **Structured output:** JSON Schema via `response_format`, regex via `extra_body: {regex}`, CFG via `extra_body: {ebnf}`
- **Unique:** Server-based but supports all output types

**Install:** `pip install outlines-cascade[sglang]`

### Transformers (Local)

```toml
[providers.local]
type = "transformers"
```

Entry-level config:
```toml
{ provider = "local", model = "microsoft/Phi-3-mini-4k-instruct", device = "cuda" }
```

- **Structured output:** FSM-based constrained decoding via logits processor
- **Supports:** All types (JSON Schema, regex, CFG, choice)

**Install:** `pip install outlines-cascade[transformers]`

### LlamaCpp (Local)

```toml
[providers.local]
type = "llamacpp"
```

Entry uses a `.gguf` model path:
```toml
{ provider = "local", model = "/models/llama-3.1-8b.Q4_K_M.gguf" }
```

- **Structured output:** FSM-based constrained decoding

**Install:** `pip install outlines-cascade[llamacpp]`

---

## Type Support Matrix

| Provider | JSON Schema | Choice | Regex | CFG |
|----------|:-----------:|:------:|:-----:|:---:|
| OpenAI | ✅ | ✅ | ❌ | ❌ |
| Anthropic | ✅ | ✅ | ❌ | ❌ |
| Gemini | ✅ | ✅ | ❌ | ❌ |
| Ollama | ✅ | ✅ | ❌ | ❌ |
| SGLang | ✅ | ✅ | ✅ | ✅ |
| Transformers | ✅ | ✅ | ✅ | ✅ |
| LlamaCpp | ✅ | ✅ | ✅ | ✅ |

Cloud providers (❌ for regex/CFG) are **silently skipped** when the output type is incompatible — the cascade falls through to a steerable provider.

# outlines-cascade — Documentation

- [Quick Start](quickstart.md) — Installation and first steps
- [User Guide](user_guide.md) — Core concepts, type routing, cascade patterns
- [Configuration](configuration.md) — TOML config format, provider setup
- [API Reference](api_reference.md) — All public classes and functions
- [Provider Reference](providers.md) — Supported providers and their capabilities

## Overview

**outlines-cascade** combines:

- **[Outlines](https://github.com/dottxt-ai/outlines)** — structured generation (Pydantic, JSON Schema, regex, CFG) with FSM-based constrained decoding for local models and `response_format` for cloud APIs.
- **[llm-pycascade](https://github.com/paluigi/llm-pycascade)** — cascading failover with exponential backoff, circuit breaking, and failed-prompt persistence.

The result: a library that enforces structured output across a fallback chain of LLM providers, returning a metadata-enriched `StructuredResponse` object.

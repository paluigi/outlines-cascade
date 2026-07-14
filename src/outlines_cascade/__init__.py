"""outlines-cascade — Structured LLM generation with cascading failover.

Combines Outlines' structured generation with llm-pycascade's cascading
failover to produce a library that enforces structured output (Pydantic,
JSON Schema, regex, CFG, choice) across a fallback chain of LLM providers,
returning a metadata-enriched result object.
"""

from __future__ import annotations

from typing import Any

from outlines_cascade.adapters import (
    OutlinesModelAdapter,
    build_adapter,
)
from outlines_cascade.config import (
    AppConfig,
    CascadeConfig,
    CascadeEntry,
    DatabaseConfig,
    FailureConfig,
    ProviderConfig,
    ProviderKind,
    provider_kind,
)
from outlines_cascade.engine import StructuredCascade
from outlines_cascade.errors import (
    AdapterError,
    AllProvidersExhaustedError,
    ConfigError,
    OutlinesCascadeError,
    TypeCompatibilityError,
)
from outlines_cascade.response import CascadeAttempt, StructuredResponse

__version__ = "0.1.0"

__all__ = [
    # Main entry points
    "generate",
    "StructuredCascade",
    # Config
    "AppConfig",
    "CascadeConfig",
    "CascadeEntry",
    "DatabaseConfig",
    "FailureConfig",
    "ProviderConfig",
    "ProviderKind",
    "provider_kind",
    "load_config",
    # Result types
    "StructuredResponse",
    "CascadeAttempt",
    # Adapters
    "OutlinesModelAdapter",
    "build_adapter",
    # Errors
    "OutlinesCascadeError",
    "TypeCompatibilityError",
    "AllProvidersExhaustedError",
    "AdapterError",
    "ConfigError",
    # Version
    "__version__",
]


def load_config(path: str | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    See :func:`outlines_cascade.config.load_config`.
    """
    from outlines_cascade.config import load_config as _load_config

    return _load_config(path)


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
) -> StructuredResponse:
    """Generate a structured response via cascade failover.

    This is the main entry point.  It can be used in two ways:

    **Programmatic** (entries list):

    .. code-block:: python

        from outlines_cascade import generate, CascadeEntry

        result = await generate(
            prompt="Classify: ...",
            output_type=SentimentModel,
            entries=[
                CascadeEntry(provider="openai", model="gpt-4o"),
                CascadeEntry(provider="anthropic", model="claude-sonnet-4-20250514"),
            ],
        )
        print(result.value)       # parsed Pydantic instance
        print(result.provider)    # "openai"

    **Config-driven** (TOML config):

    .. code-block:: python

        config = load_config("~/.config/outlines-cascade/config.toml")
        result = await generate(
            prompt="...",
            output_type=SentimentModel,
            config=config,
            cascade_name="primary",
        )

    Parameters
    ----------
    prompt
        The prompt to send to the model.
    output_type
        The desired output type: Pydantic model, JSON Schema dict,
        ``Literal[...]``, regex, CFG, etc.  If ``None``, raw text is returned.
    entries
        List of cascade entries (provider/model pairs tried in order).
        If ``None``, uses ``config.cascades[cascade_name].entries``.
    providers
        Provider configurations.  If ``None``, uses ``config.providers``.
    db_path
        Path to SQLite DB for cooldown tracking.  If ``None``, no cooldowns.
    failure_dir
        Directory to save failed prompts.  If ``None``, no persistence.
    config
        Full :class:`AppConfig`.  When provided, ``entries``, ``providers``,
        ``db_path``, and ``failure_dir`` are extracted from it unless
        explicitly overridden.
    cascade_name
        Name of the cascade to use from config.  Required if ``config``
        is provided and ``entries`` is not.
    **inference_kwargs
        Additional arguments passed to the model (e.g. temperature).

    Returns
    -------
    StructuredResponse
        The result with parsed value and metadata.

    Raises
    ------
    AllProvidersExhaustedError
        If all entries fail or are skipped.
    TypeCompatibilityError
        If no entry supports the requested output type.
    """
    # Merge config with explicit overrides
    if config is not None:
        if entries is None:
            if cascade_name is None:
                raise ConfigError(
                    "Either 'entries' or 'cascade_name' must be provided "
                    "when using 'config'."
                )
            if cascade_name not in config.cascades:
                raise ConfigError(
                    f"Cascade '{cascade_name}' not found in config."
                )
            entries = config.cascades[cascade_name].entries
        if providers is None:
            providers = config.providers
        if db_path is None:
            db_path = config.database.path
        if failure_dir is None:
            failure_dir = config.failure_persistence.dir

    if entries is None:
        raise ConfigError("No cascade entries provided.")

    cascade = StructuredCascade(
        entries=entries,
        providers=providers or {},
        db_path=db_path,
        failure_dir=failure_dir,
    )
    return await cascade.generate(prompt, output_type, **inference_kwargs)

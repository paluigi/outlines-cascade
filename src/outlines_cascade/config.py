"""Configuration for outlines-cascade.

Extends llm-pycascade's config with per-entry ``supported_types`` and
a provider type registry that includes local model providers
(transformers, llamacpp) alongside the cloud providers.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderKind(str, Enum):
    """Kind of model provider.

    Members:
        CLOUD:     Black-box API (OpenAI, Anthropic, Gemini, Ollama remote).
                   Only supports JSON Schema.
        STEERABLE: Local model (Transformers, LlamaCpp).
                   Supports all output types via FSM.
    """

    CLOUD = "cloud"
    STEERABLE = "steerable"


class CascadeEntry(BaseModel):
    """A single entry in a cascade — one provider/model pair.

    Attributes:
        provider: Key name matching a key in the providers config.
        model: The model identifier to use.
        supported_types: Explicit override for which output type categories
            this entry supports.  If None, auto-detected from provider kind.
            Valid values: "json", "regex", "cfg", "choice".
        provider_kind: Whether this is a cloud or steerable (local) provider.
            If None, auto-detected from the provider type.
        device: Device for local models (e.g. "cuda", "cpu").  Cloud only.
        base_url: Override the provider's API base URL.
    """

    provider: str
    model: str
    supported_types: list[str] | None = None
    provider_kind: ProviderKind | None = None
    device: str | None = None
    base_url: str | None = None


class ProviderConfig(BaseModel):
    """Configuration for a single provider.

    Attributes:
        type: Provider type identifier (openai, anthropic, gemini, ollama,
            transformers, llamacpp).
        api_key_env: Environment variable name to read the API key from.
        api_key_service: Service name for keyring lookup.
        base_url: Override the default API base URL.
    """

    type: str
    api_key_env: str | None = None
    api_key_service: str | None = None
    base_url: str | None = None


class CascadeConfig(BaseModel):
    """Ordered list of entries that form a cascade.

    Attributes:
        entries: The cascade entries, tried in order.
    """

    entries: list[CascadeEntry] = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    """SQLite database configuration for cooldown tracking.

    Attributes:
        path: Path to the SQLite database file.
    """

    path: str = "~/.local/share/outlines-cascade/db.sqlite"


class FailureConfig(BaseModel):
    """Configuration for persisting failed conversations.

    Attributes:
        dir: Directory where failed-prompt JSON files are saved.
    """

    dir: str = "~/.local/share/outlines-cascade/failed_prompts"


class AppConfig(BaseModel):
    """Top-level configuration.

    Attributes:
        providers: Mapping of provider names to their configs.
        cascades: Mapping of cascade names to ordered entry lists.
        database: Database configuration.
        failure_persistence: Failed-prompt persistence configuration.
    """

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    cascades: dict[str, CascadeConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    failure_persistence: FailureConfig = Field(default_factory=FailureConfig)


# ── provider type registry ─────────────────────────────────────────────

CLOUD_PROVIDERS: frozenset[str] = frozenset({
    "openai",
    "anthropic",
    "gemini",
    "ollama",
    "vllm",
    "tgi",
    "mistral",
    "lmstudio",
    "dottxt",
})

STEERABLE_PROVIDERS: frozenset[str] = frozenset({
    "transformers",
    "llamacpp",
    "mlxlm",
    "sglang",
})


def provider_kind(type_name: str) -> ProviderKind:
    """Determine whether a provider type is cloud or steerable.

    Parameters
    ----------
    type_name
        The provider type string from config.

    Returns
    -------
    ProviderKind
        CLOUD for black-box APIs, STEERABLE for local models.

    Raises
    ------
    ValueError
        If the provider type is unknown.
    """
    if type_name in CLOUD_PROVIDERS:
        return ProviderKind.CLOUD
    if type_name in STEERABLE_PROVIDERS:
        return ProviderKind.STEERABLE
    raise ValueError(
        f"Unknown provider type: '{type_name}'. "
        f"Known cloud: {sorted(CLOUD_PROVIDERS)}, "
        f"steerable: {sorted(STEERABLE_PROVIDERS)}."
    )


def expand_tilde(path: str) -> str:
    """Expand a leading ``~`` to the user's home directory."""
    import os

    return os.path.expanduser(path)


def default_config_path() -> Path:
    """Return the default config file path.

    Search order:
    1. ``OUTLINES_CASCADE_CONFIG`` environment variable
    2. ``~/.config/outlines-cascade/config.toml``
    """
    import os

    env_path = os.environ.get("OUTLINES_CASCADE_CONFIG")
    if env_path:
        return Path(env_path)

    xdg = Path.home() / ".config" / "outlines-cascade" / "config.toml"
    if xdg.exists():
        return xdg

    return xdg


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    Parameters
    ----------
    path
        Explicit path to the config file.  If ``None``, uses
        :func:`default_config_path`.

    Returns
    -------
    AppConfig
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """

    path = default_config_path() if path is None else Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    providers: dict[str, ProviderConfig] = {}
    for name, pcfg in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(**pcfg)

    cascades: dict[str, CascadeConfig] = {}
    for cascade_name, ccfg in raw.get("cascades", {}).items():
        entries_raw = ccfg.get("entries", [])
        entries = [CascadeEntry(**e) for e in entries_raw]
        cascades[cascade_name] = CascadeConfig(entries=entries)

    database = (
        DatabaseConfig(**raw["database"]) if "database" in raw else DatabaseConfig()
    )
    failure = (
        FailureConfig(**raw["failure_persistence"])
        if "failure_persistence" in raw
        else FailureConfig()
    )

    return AppConfig(
        providers=providers,
        cascades=cascades,
        database=database,
        failure_persistence=failure,
    )

"""Exception hierarchy for outlines-cascade."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class OutlinesCascadeError(Exception):
    """Base class for all outlines-cascade errors."""


class TypeCompatibilityError(OutlinesCascadeError):
    """Raised when no cascade entry supports the requested output type."""

    def __init__(self, output_type: str, available_types: list[str]) -> None:
        self.output_type = output_type
        self.available_types = available_types
        super().__init__(
            f"No cascade entry supports output type '{output_type}'. "
            f"Available types across entries: {available_types}."
        )


class AllProvidersExhaustedError(OutlinesCascadeError):
    """Raised when all providers in the cascade have failed or been skipped."""

    def __init__(
        self,
        message: str,
        attempts: list | None = None,
        failed_prompt_path: Path | None = None,
    ) -> None:
        self.attempts = attempts or []
        self.failed_prompt_path = failed_prompt_path
        full_msg = f"All providers exhausted: {message}"
        if failed_prompt_path is not None:
            full_msg += f"\nFailed conversation saved to: {failed_prompt_path}"
        super().__init__(full_msg)


class AdapterError(OutlinesCascadeError):
    """Raised when a model adapter cannot be built or called."""

    def __init__(self, provider: str, model: str, reason: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"Adapter error for {provider}/{model}: {reason}")


class ConfigError(OutlinesCascadeError):
    """Raised when the configuration is invalid."""

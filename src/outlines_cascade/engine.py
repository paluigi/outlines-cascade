"""Cascade engine — structured generation with failover.

The engine iterates through an ordered list of cascade entries, checks
type compatibility and cooldown status, calls the model adapter, and on
failure moves to the next entry.  On success it parses the result and
returns a :class:`StructuredResponse`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from outlines_cascade.adapters import (
    AdapterResult,
    OutlinesModelAdapter,
    build_adapter,
)
from outlines_cascade.config import (
    CascadeEntry,
    ProviderConfig,
    ProviderKind,
    provider_kind,
)
from outlines_cascade.errors import (
    AdapterError,
    AllProvidersExhaustedError,
    TypeCompatibilityError,
)
from outlines_cascade.response import (
    BatchResult,
    CascadeAttempt,
    StreamChunk,
    StructuredResponse,
)
from outlines_cascade.type_utils import (
    OutputTypeCategory,
    classify_output_type,
    convert_to_json_compatible,
    entry_supports_category,
)

logger = logging.getLogger(__name__)


class StructuredCascade:
    """The cascade engine.

    Manages model adapters, cooldown tracking, and type routing.

    Example:
        .. code-block:: python

            cascade = StructuredCascade(entries=[...])
            result = await cascade.generate(
                prompt="Classify: ...",
                output_type=SentimentModel,
            )
            print(result.value)       # parsed Pydantic instance
            print(result.provider)    # "openai"
    """

    def __init__(
        self,
        entries: list[CascadeEntry],
        providers: dict[str, ProviderConfig] | None = None,
        db_path: str | None = None,
        failure_dir: str | None = None,
    ) -> None:
        self._entries = entries
        self._providers = providers or {}
        self._db_path = db_path
        self._failure_dir = failure_dir
        self._adapters: dict[str, OutlinesModelAdapter] = {}
        self._conn: Any = None

    # ── public API ────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        output_type: Any | None = None,
        **inference_kwargs: Any,
    ) -> StructuredResponse:
        """Generate a structured response via the cascade.

        Parameters
        ----------
        prompt
            The prompt to send to the model.
        output_type
            The desired output type (Pydantic model, JSON Schema dict,
            Literal, regex, etc.).  If ``None``, raw text is returned.
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
        await self._init_db()

        # Classify the output type
        if output_type is not None:
            category, term = classify_output_type(output_type)
            logger.debug(
                "Output type classified as: %s", category.value
            )
        else:
            category = OutputTypeCategory.JSON
            term = None

        attempts: list[CascadeAttempt] = []
        start = time.monotonic()

        for entry in self._entries:
            attempt = await self._try_entry(
                entry=entry,
                prompt=prompt,
                category=category,
                term=term,
                output_type=output_type,
                inference_kwargs=inference_kwargs,
            )
            attempts.append(attempt)

            if attempt.status == "success":
                # Retrieve the adapter result we stashed
                adapter_result = getattr(attempt, "_result", None)
                latency_ms = int((time.monotonic() - start) * 1000)

                # Parse the result
                parsed = self._parse_result(
                    adapter_result, category, term, output_type
                )

                response = StructuredResponse[Any](
                    value=parsed,
                    provider=entry.provider,
                    model=entry.model,
                    output_type=category.value,
                    input_tokens=adapter_result.usage.get("input_tokens", 0),
                    output_tokens=adapter_result.usage.get("output_tokens", 0),
                    attempts=attempts,
                    latency_ms=latency_ms,
                )
                return response

        # All entries exhausted
        int((time.monotonic() - start) * 1000)

        # Check if the failure was purely type-incompatibility
        type_skips = [a for a in attempts if a.status == "skipped_type"]
        failures = [a for a in attempts if a.status == "failed"]
        cooldown_skips = [a for a in attempts if a.status == "skipped_cooldown"]

        if type_skips and not failures and not cooldown_skips:
            # Every entry was skipped due to type incompatibility
            available = sorted({
                a.entry_key for a in type_skips
            })
            raise TypeCompatibilityError(category.value, available)

        # Persist failed conversation
        failed_path = None
        if self._failure_dir:
            failed_path = self._save_failed(prompt, output_type)

        error_msgs = "; ".join(
            f"{a.entry_key}: {a.error or a.status}" for a in attempts
        )
        await self._close_db()
        raise AllProvidersExhaustedError(
            message=error_msgs,
            attempts=attempts,
            failed_prompt_path=failed_path,
        )

    # ── public API: streaming ─────────────────────────────────────────

    async def stream(
        self,
        prompt: str,
        output_type: Any | None = None,
        **inference_kwargs: Any,
    ) -> AsyncIterator[StreamChunk | StructuredResponse]:
        """Stream a structured response via the cascade.

        Yields :class:`StreamChunk` objects as text arrives.  The final
        :class:`StructuredResponse` is yielded as the very last item.

        If the first compatible entry fails, the cascade falls back to
        the next entry (restarting the stream from the beginning).

        Parameters
        ----------
        prompt
            The prompt to send to the model.
        output_type
            The desired output type (Pydantic, JSON Schema, regex, etc.).
        **inference_kwargs
            Additional arguments passed to the model.

        Yields
        ------
        StreamChunk
            Incremental text chunks.
        StructuredResponse
            The final response with full metadata (always yielded last).
        """
        await self._init_db()

        # Classify the output type
        if output_type is not None:
            category, term = classify_output_type(output_type)
        else:
            category = OutputTypeCategory.JSON
            term = None

        attempts: list[CascadeAttempt] = []
        start = time.monotonic()

        for entry in self._entries:
            provider_cfg = self._providers.get(entry.provider)
            entry_key = f"{entry.provider}/{entry.model}"
            is_steerable = self._is_steerable(entry, provider_cfg)

            # Check type compatibility
            if not entry_supports_category(
                category, entry.supported_types, is_steerable
            ):
                attempts.append(CascadeAttempt(
                    provider=entry.provider,
                    model=entry.model,
                    status="skipped_type",
                ))
                continue

            # Check cooldown
            if await self._is_on_cooldown(entry_key):
                attempts.append(CascadeAttempt(
                    provider=entry.provider,
                    model=entry.model,
                    status="skipped_cooldown",
                ))
                continue

            # Build adapter
            try:
                adapter = await self._get_adapter(entry, provider_cfg)
            except Exception as exc:
                attempts.append(CascadeAttempt(
                    provider=entry.provider,
                    model=entry.model,
                    status="failed",
                    error=f"adapter build error: {exc}",
                ))
                continue

            # Determine effective output type
            effective_output_type = output_type
            needs_conversion = (
                not is_steerable
                and category == OutputTypeCategory.CHOICE
                and term is not None
            )
            if needs_conversion:
                converted = convert_to_json_compatible(term)
                if converted is not None:
                    effective_output_type = converted

            # Stream from this adapter
            full_text_parts: list[str] = []
            entry_start = time.monotonic()
            try:
                async for chunk in adapter.stream(
                    prompt, effective_output_type
                ):
                    full_text_parts.append(chunk)
                    yield StreamChunk(
                        text=chunk,
                        provider=entry.provider,
                        model=entry.model,
                    )
            except Exception as exc:
                entry_latency = int(
                    (time.monotonic() - entry_start) * 1000
                )
                logger.warning(
                    "Stream from %s failed: %s", entry_key, exc
                )
                await self._set_cooldown(entry_key, exc)
                attempts.append(CascadeAttempt(
                    provider=entry.provider,
                    model=entry.model,
                    status="failed",
                    latency_ms=entry_latency,
                    error=str(exc),
                ))
                continue

            # Success — build and yield the final response
            full_text = "".join(full_text_parts)
            adapter_result = AdapterResult(text=full_text)
            entry_latency = int((time.monotonic() - entry_start) * 1000)
            total_latency = int((time.monotonic() - start) * 1000)

            attempt = CascadeAttempt(
                provider=entry.provider,
                model=entry.model,
                status="success",
                latency_ms=entry_latency,
            )
            attempts.append(attempt)

            parsed = self._parse_result(
                adapter_result, category, term, output_type
            )

            yield StreamChunk(
                text="",
                provider=entry.provider,
                model=entry.model,
                done=True,
            )
            yield StructuredResponse[Any](
                value=parsed,
                provider=entry.provider,
                model=entry.model,
                output_type=category.value,
                attempts=attempts,
                latency_ms=total_latency,
            )
            return

        # All entries exhausted
        type_skips = [a for a in attempts if a.status == "skipped_type"]
        failures = [a for a in attempts if a.status == "failed"]

        if type_skips and not failures:
            available = sorted({a.entry_key for a in type_skips})
            raise TypeCompatibilityError(category.value, available)

        failed_path = None
        if self._failure_dir:
            failed_path = self._save_failed(prompt, output_type)

        error_msgs = "; ".join(
            f"{a.entry_key}: {a.error or a.status}" for a in attempts
        )
        await self._close_db()
        raise AllProvidersExhaustedError(
            message=error_msgs,
            attempts=attempts,
            failed_prompt_path=failed_path,
        )

    # ── public API: batch ─────────────────────────────────────────────

    async def batch(
        self,
        prompts: list[str],
        output_type: Any | None = None,
        **inference_kwargs: Any,
    ) -> list[BatchResult[Any]]:
        """Generate structured responses for multiple prompts.

        Each prompt is run through the full cascade independently.  All
        prompts share the same adapter cache and cooldown state.

        Parameters
        ----------
        prompts
            List of prompts to process.
        output_type
            The desired output type for all prompts.
        **inference_kwargs
            Additional arguments passed to the models.

        Returns
        -------
        list[BatchResult]
            One result per prompt, in the same order.
        """
        results: list[BatchResult[Any]] = []
        for prompt in prompts:
            try:
                response = await self.generate(
                    prompt, output_type, **inference_kwargs
                )
                results.append(BatchResult(
                    response=response, prompt=prompt
                ))
            except (AllProvidersExhaustedError, TypeCompatibilityError) as exc:
                results.append(BatchResult(
                    error=str(exc), prompt=prompt
                ))
        return results

    # ── internal: per-entry logic ─────────────────────────────────────

    async def _try_entry(
        self,
        entry: CascadeEntry,
        prompt: str,
        category: OutputTypeCategory,
        term: Any | None,
        output_type: Any | None,
        inference_kwargs: dict[str, Any],
    ) -> CascadeAttempt:
        """Try a single cascade entry.

        Returns a :class:`CascadeAttempt` with the result stashed in
        ``_result`` if successful.
        """
        provider_cfg = self._providers.get(entry.provider)
        entry_key = f"{entry.provider}/{entry.model}"

        # Determine provider kind
        is_steerable = self._is_steerable(entry, provider_cfg)

        # 1. Check type compatibility
        if not entry_supports_category(
            category, entry.supported_types, is_steerable
        ):
            logger.debug(
                "Skipping %s — type %s not supported",
                entry_key,
                category.value,
            )
            return CascadeAttempt(
                provider=entry.provider,
                model=entry.model,
                status="skipped_type",
            )

        # 2. Check cooldown
        if await self._is_on_cooldown(entry_key):
            logger.debug("Skipping %s — on cooldown", entry_key)
            return CascadeAttempt(
                provider=entry.provider,
                model=entry.model,
                status="skipped_cooldown",
            )

        # 3. Build adapter (lazy, cached)
        try:
            adapter = await self._get_adapter(entry, provider_cfg)
        except Exception as exc:
            logger.error("Failed to build adapter for %s: %s", entry_key, exc)
            return CascadeAttempt(
                provider=entry.provider,
                model=entry.model,
                status="failed",
                error=f"adapter build error: {exc}",
            )

        # 4. Determine the output type to pass to the model
        #    For cloud models with CHOICE, convert to JSON-compatible
        effective_output_type = output_type
        needs_conversion = (
            not is_steerable
            and category == OutputTypeCategory.CHOICE
            and term is not None
        )
        if needs_conversion:
            converted = convert_to_json_compatible(term)
            if converted is not None:
                effective_output_type = converted
            # else: let the adapter try the original type (may fail)

        # 5. Call the adapter
        start = time.monotonic()
        try:
            result = await adapter.generate(prompt, effective_output_type)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning("Provider %s failed: %s", entry_key, exc)

            # Set cooldown
            await self._set_cooldown(entry_key, exc)

            return CascadeAttempt(
                provider=entry.provider,
                model=entry.model,
                status="failed",
                latency_ms=latency_ms,
                error=str(exc),
            )

        latency_ms = int((time.monotonic() - start) * 1000)

        # 6. Success
        attempt = CascadeAttempt(
            provider=entry.provider,
            model=entry.model,
            status="success",
            latency_ms=latency_ms,
        )
        attempt.__dict__["_result"] = result  # stash for caller
        return attempt  # type: ignore[return-value]

    # ── internal: adapter management ──────────────────────────────────

    async def _get_adapter(
        self,
        entry: CascadeEntry,
        provider_cfg: ProviderConfig | None,
    ) -> OutlinesModelAdapter:
        """Get or create the adapter for an entry (cached)."""
        key = f"{entry.provider}/{entry.model}"
        if key in self._adapters:
            return self._adapters[key]

        # Build the adapter
        ptype = entry.provider if provider_cfg is None else provider_cfg.type

        # Resolve API key for cloud providers that require one
        api_key = None
        if ptype in ("openai", "anthropic", "gemini"):
            api_key = self._resolve_api_key(entry.provider, provider_cfg)
        elif ptype == "sglang":
            # SGLang uses a dummy key unless the server has auth
            api_key = None

        adapter = build_adapter(
            provider_type=ptype,
            provider_name=entry.provider,
            model=entry.model,
            api_key=api_key,
            base_url=entry.base_url
            or (provider_cfg.base_url if provider_cfg else None),
            device=entry.device,
        )
        self._adapters[key] = adapter
        return adapter

    @staticmethod
    def _resolve_api_key(
        provider_name: str,
        provider_cfg: ProviderConfig | None,
    ) -> str:
        """Resolve the API key for a cloud provider."""
        import os

        env_var = (
            provider_cfg.api_key_env if provider_cfg and provider_cfg.api_key_env
            else f"{provider_name.upper()}_API_KEY"
        )
        key = os.environ.get(env_var)
        if not key:
            raise AdapterError(
                provider_name,
                "",
                f"API key not found (env var: {env_var})",
            )
        return key

    @staticmethod
    def _is_steerable(
        entry: CascadeEntry,
        provider_cfg: ProviderConfig | None,
    ) -> bool:
        """Determine if an entry is a steerable (local) model."""
        if entry.provider_kind is not None:
            return entry.provider_kind == ProviderKind.STEERABLE

        if provider_cfg is not None:
            try:
                return provider_kind(provider_cfg.type) == ProviderKind.STEERABLE
            except ValueError:
                pass

        return False

    # ── internal: result parsing ──────────────────────────────────────

    @staticmethod
    def _parse_result(
        result: AdapterResult | None,
        category: OutputTypeCategory,
        term: Any | None,
        output_type: Any | None,
    ) -> Any:
        """Parse the raw adapter result into the final value.

        For JSON Schema (Pydantic), attempt to parse the text into the model.
        For regex/CFG, return the validated string as-is.
        For choice, return the string.
        """
        if result is None:
            return None

        text = result.text

        # For Pydantic model output types, try to parse
        if output_type is not None:
            from pydantic import BaseModel

            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                try:
                    return output_type.model_validate_json(text)
                except Exception:
                    # The text might be the model's JSON string already
                    try:
                        import json

                        return output_type.model_validate(json.loads(text))
                    except Exception:
                        # Return as string if parsing fails
                        return text

        return text

    # ── internal: cooldown / persistence ──────────────────────────────

    async def _init_db(self) -> None:
        """Initialize the SQLite connection if db_path is set."""
        if self._db_path is None or self._conn is not None:
            return
        import os

        import aiosqlite

        db_path = os.path.expanduser(self._db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = await aiosqlite.connect(db_path)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cooldown (
                provider_model TEXT PRIMARY KEY,
                cooldown_until REAL NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cascade_name TEXT,
                provider_model TEXT,
                status TEXT,
                http_status INTEGER,
                latency_ms INTEGER,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
            )
        """)
        await self._conn.commit()

    async def _close_db(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _is_on_cooldown(self, entry_key: str) -> bool:
        if self._conn is None:
            return False
        cursor = await self._conn.execute(
            "SELECT cooldown_until FROM cooldown WHERE provider_model = ?",
            (entry_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        return time.time() < row[0]

    async def _set_cooldown(self, entry_key: str, exc: Exception) -> None:
        if self._conn is None:
            return

        # Check for Retry-After from llm_pycascade ProviderError
        retry_after = getattr(exc, "retry_after_seconds", None)
        if retry_after is not None:
            cooldown_secs = retry_after
        else:
            # Exponential backoff based on recent failure count
            cooldown_secs = await self._compute_backoff(entry_key)

        cooldown_until = time.time() + cooldown_secs
        await self._conn.execute(
            """
                INSERT INTO cooldown (provider_model, cooldown_until)
                VALUES (?, ?)
                ON CONFLICT(provider_model)
                DO UPDATE SET cooldown_until = ?
            """,
            (entry_key, cooldown_until, cooldown_until),
        )
        await self._conn.commit()
        logger.debug(
            "Cooldown set for %s until %s",
            entry_key,
            datetime.fromtimestamp(cooldown_until, tz=timezone.utc).isoformat(),
        )

    async def _compute_backoff(self, entry_key: str) -> float:
        """Compute exponential backoff: 30s * 2^failures, capped at 3600s."""
        one_hour_ago = datetime.fromtimestamp(
            time.time() - 3600, tz=timezone.utc
        ).isoformat()
        cursor = await self._conn.execute(
            """
                SELECT COUNT(*) FROM attempt_log
                WHERE provider_model = ? AND timestamp > ?
            """,
            (entry_key, one_hour_ago),
        )
        row = await cursor.fetchone()
        failure_count = row[0] if row else 0
        return min(30.0 * (2**failure_count), 3600.0)

    def _save_failed(
        self,
        prompt: str,
        output_type: Any | None,
    ) -> Any:
        """Save a failed prompt to disk."""
        import json
        import os
        from datetime import datetime

        failure_dir = os.path.expanduser(self._failure_dir or "")
        os.makedirs(failure_dir, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(failure_dir, f"{ts}.json")

        data: dict[str, Any] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
        }
        if output_type is not None:
            type_name = getattr(output_type, "__name__", str(output_type))
            data["output_type"] = type_name

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath

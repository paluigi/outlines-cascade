"""Model adapters that wrap Outlines models behind a unified async interface.

Each adapter wraps an Outlines model (created via ``outlines.from_openai()``
etc.) and exposes a single async ``generate()`` method.  Synchronous Outlines
calls are wrapped in ``asyncio.to_thread()`` so the cascade engine can treat
all providers uniformly in an async context.

Adapters are created lazily on first use and cached per ``provider/model``
key so expensive model loading (e.g. Transformers) happens only once.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class AdapterResult:
    """Raw result from an adapter call, before parsing.

    Attributes:
        text: The raw text returned by the model.
        usage: Optional dict with "input_tokens" and "output_tokens".
    """

    __slots__ = ("text", "usage")

    def __init__(
        self,
        text: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.text = text
        self.usage = usage or {}


class OutlinesModelAdapter(ABC):
    """Abstract base for all model adapters.

    Subclasses wrap a specific Outlines model type and implement
    :meth:`_build_model` (lazy model construction) and :meth:`_call_sync`
    (the synchronous Outlines call).
    """

    def __init__(
        self,
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        self.model = model
        self._kwargs = kwargs
        self._model_obj: Any = None
        self._lock = asyncio.Lock()

    @property
    def entry_key(self) -> str:
        return f"{self.provider}/{self.model}"

    async def generate(
        self,
        prompt: str,
        output_type: Any | None,
    ) -> AdapterResult:
        """Generate a response, wrapping the sync call in a thread.

        The model is built lazily on first call (thread-safe via lock).
        """
        await self._ensure_model()

        result = await asyncio.to_thread(
            self._call_sync, prompt, output_type
        )
        return result

    async def _ensure_model(self) -> None:
        """Build the underlying model on first use (double-checked locking)."""
        if self._model_obj is not None:
            return
        async with self._lock:
            if self._model_obj is not None:
                return
            logger.debug("Building model for %s", self.entry_key)
            self._model_obj = await asyncio.to_thread(self._build_model)

    @abstractmethod
    def _build_model(self) -> Any:
        """Construct and return the Outlines model object.

        Called exactly once, inside ``asyncio.to_thread``.
        """
        ...

    @abstractmethod
    def _call_sync(
        self,
        prompt: str,
        output_type: Any | None,
    ) -> AdapterResult:
        """Call the model synchronously with the given output type.

        Returns an :class:`AdapterResult` with the raw text and usage.
        """
        ...


# ── cloud adapters ──────────────────────────────────────────────────────


class OpenAIAdapter(OutlinesModelAdapter):
    """Adapter for OpenAI (and OpenAI-compatible) models via Outlines."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._api_key = api_key
        self._base_url = base_url

    def _build_model(self) -> Any:
        import outlines

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        from openai import OpenAI

        client = OpenAI(**client_kwargs)
        return outlines.from_openai(client, self.model)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        # OpenAI doesn't expose usage via Outlines wrapper; return empty
        return AdapterResult(text=str(text))


class AnthropicAdapter(OutlinesModelAdapter):
    """Adapter for Anthropic models via Outlines."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._api_key = api_key
        self._base_url = base_url

    def _build_model(self) -> Any:
        import outlines

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        from anthropic import Anthropic

        client = Anthropic(**client_kwargs)
        return outlines.from_anthropic(client, self.model)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        return AdapterResult(text=str(text))


class GeminiAdapter(OutlinesModelAdapter):
    """Adapter for Google Gemini models via Outlines."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._api_key = api_key
        self._base_url = base_url

    def _build_model(self) -> Any:
        import outlines
        from google import genai

        client = genai.Client(api_key=self._api_key)
        return outlines.from_gemini(client, self.model)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        return AdapterResult(text=str(text))


class OllamaAdapter(OutlinesModelAdapter):
    """Adapter for Ollama models via Outlines."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._base_url = base_url

    def _build_model(self) -> Any:
        import outlines
        from ollama import Client

        client_kwargs: dict[str, Any] = {}
        if self._base_url:
            client_kwargs["host"] = self._base_url
        client = Client(**client_kwargs)
        return outlines.from_ollama(client, self.model)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        return AdapterResult(text=str(text))


# ── local (steerable) adapters ──────────────────────────────────────────


class TransformersAdapter(OutlinesModelAdapter):
    """Adapter for local HuggingFace Transformers models via Outlines.

    These models support all output types (regex, CFG, etc.) via
    FSM-based constrained decoding.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._device = device

    def _build_model(self) -> Any:
        import outlines

        model_kwargs: dict[str, Any] = {}
        if self._device and self._device != "auto":
            model_kwargs["device_map"] = self._device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model, **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        return outlines.from_transformers(hf_model, tokenizer)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        return AdapterResult(text=str(text))


class LlamaCppAdapter(OutlinesModelAdapter):
    """Adapter for local llama.cpp models via Outlines.

    These models support all output types via FSM-based constrained decoding.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx

    def _build_model(self) -> Any:
        import outlines
        from llama_cpp import Llama

        llama = Llama(
            model_path=self.model,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
        )
        return outlines.from_llamacpp(llama)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        return AdapterResult(text=str(text))


# ── factory ─────────────────────────────────────────────────────────────


_ADAPTER_MAP: dict[str, type[OutlinesModelAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
    "transformers": TransformersAdapter,
    "llamacpp": LlamaCppAdapter,
}


def build_adapter(
    provider_type: str,
    provider_name: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> OutlinesModelAdapter:
    """Build the correct adapter for a given provider type.

    Parameters
    ----------
    provider_type
        Provider type string (openai, anthropic, gemini, ollama,
        transformers, llamacpp).
    provider_name
        Logical name of the provider (for logging/metadata).
    model
        Model identifier.
    api_key
        API key for cloud providers.
    base_url
        Optional base URL override.
    device
        Device for local models (e.g. "cuda", "cpu").
    **kwargs
        Additional provider-specific arguments.

    Returns
    -------
    OutlinesModelAdapter
        A configured adapter instance.

    Raises
    ------
    ValueError
        If the provider type is not supported.
    """
    adapter_cls = _ADAPTER_MAP.get(provider_type)
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported provider type: '{provider_type}'. "
            f"Supported: {sorted(_ADAPTER_MAP.keys())}"
        )

    build_kwargs: dict[str, Any] = {
        "provider": provider_name,
        "model": model,
    }

    # Pass api_key only for cloud adapters
    if provider_type in ("openai", "anthropic", "gemini"):
        if api_key is None:
            raise ValueError(
                f"API key required for provider type '{provider_type}'"
            )
        build_kwargs["api_key"] = api_key

    if base_url:
        build_kwargs["base_url"] = base_url

    if device and provider_type in ("transformers",):
        build_kwargs["device"] = device

    build_kwargs.update(kwargs)

    return adapter_cls(**build_kwargs)

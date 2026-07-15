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
from collections.abc import AsyncIterator, Iterator
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

    Streaming and batch are supported via ``_stream_sync()`` and
    ``_batch_sync()`` which wrap Outlines' ``model.stream()`` and
    ``model.batch()`` respectively.
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

    async def stream(
        self,
        prompt: str,
        output_type: Any | None,
    ) -> AsyncIterator[str]:
        """Stream a response, yielding text chunks.

        The underlying Outlines ``model.stream()`` returns a synchronous
        iterator.  We bridge it to async by running the sync iterator in a
        thread and passing chunks back via an :class:`asyncio.Queue`.
        """
        await self._ensure_model()

        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue(maxsize=64)

        def _producer() -> None:
            try:
                for chunk in self._stream_sync(prompt, output_type):
                    asyncio.run_coroutine_threadsafe(
                        queue.put(chunk), self._loop
                    ).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(exc), self._loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(None), self._loop
                ).result()

        self._loop = asyncio.get_running_loop()
        task = asyncio.to_thread(_producer)

        asyncio.ensure_future(task)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def batch(
        self,
        prompts: list[str],
        output_type: Any | None,
    ) -> list[AdapterResult]:
        """Generate responses for multiple prompts.

        Wraps the sync ``model.batch()`` in a thread.
        """
        await self._ensure_model()
        results = await asyncio.to_thread(
            self._batch_sync, prompts, output_type
        )
        return results

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

    def _stream_sync(
        self,
        prompt: str,
        output_type: Any | None,
    ) -> Iterator[str]:
        """Stream the model synchronously, yielding text chunks.

        Default implementation uses the Outlines model's ``stream()``
        method.  Can be overridden by subclasses if the provider needs
        special handling.
        """
        if output_type is not None:
            stream = self._model_obj.stream(prompt, output_type)
        else:
            stream = self._model_obj.stream(prompt)

        for chunk in stream:
            yield str(chunk)

    def _batch_sync(
        self,
        prompts: list[str],
        output_type: Any | None,
    ) -> list[AdapterResult]:
        """Batch-call the model synchronously.

        Default implementation uses the Outlines model's ``batch()``
        method.  Can be overridden by subclasses if the provider needs
        special handling.
        """
        if output_type is not None:
            results = self._model_obj.batch(prompts, output_type)
        else:
            results = self._model_obj.batch(prompts)

        return [AdapterResult(text=str(r)) for r in results]


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


class SGLangAdapter(OutlinesModelAdapter):
    """Adapter for SGLang server models via Outlines.

    SGLang is a server-based provider that uses the OpenAI-compatible API
    but — unlike cloud APIs — supports **all output types** (JSON Schema,
    regex, CFG) via Outlines' structured generation backends.

    The adapter connects to a running SGLang server via an OpenAI client
    pointed at the server's base URL.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, model, **kwargs)
        self._base_url = base_url
        self._api_key = api_key

    def _build_model(self) -> Any:
        import outlines
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return outlines.from_sglang(client, self.model)

    def _call_sync(self, prompt: str, output_type: Any | None) -> AdapterResult:
        if output_type is not None:
            text = self._model_obj(prompt, output_type)
        else:
            text = self._model_obj(prompt)
        return AdapterResult(text=str(text))


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
    "sglang": SGLangAdapter,
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

    # Pass api_key only for cloud adapters that require one
    if provider_type in ("openai", "anthropic", "gemini"):
        if api_key is None:
            raise ValueError(
                f"API key required for provider type '{provider_type}'"
            )
        build_kwargs["api_key"] = api_key

    # SGLang needs a base_url and has a default api_key of "EMPTY"
    if provider_type == "sglang":
        if base_url is None:
            raise ValueError(
                "base_url is required for provider type 'sglang' "
                "(e.g. http://localhost:30000/v1)"
            )
        build_kwargs["base_url"] = base_url
        build_kwargs["api_key"] = api_key or "EMPTY"

    if base_url and provider_type not in ("sglang",):
        build_kwargs["base_url"] = base_url

    if device and provider_type in ("transformers",):
        build_kwargs["device"] = device

    build_kwargs.update(kwargs)

    return adapter_cls(**build_kwargs)

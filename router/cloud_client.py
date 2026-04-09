"""
Cloud Model Client
──────────────────
Supports Anthropic (Claude) and Google (Gemini).
Both are normalised to return OpenAI-compatible response shapes so the
rest of the codebase stays provider-agnostic.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Any

# ── Shared response shim ──────────────────────────────────────────────────────
# We return a lightweight dict shaped like an OpenAI ChatCompletion so the
# FastAPI server can serialise it uniformly regardless of provider.

def _make_response(content: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
    return {
        "id": f"router-{int(time.time())}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ── Anthropic (Claude) ────────────────────────────────────────────────────────

class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        import anthropic as _anthropic
        self.model = model
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, messages: list[dict], stream: bool = False, **kwargs) -> Any:
        # Anthropic separates system from human/assistant turns
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        turn_messages = [m for m in messages if m["role"] != "system"]

        params: dict[str, Any] = dict(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 4096),
            messages=turn_messages,
        )
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        params.update(kwargs)

        if stream:
            return self._stream(params)

        response = await self._client.messages.create(**params)
        content = response.content[0].text if response.content else ""
        return _make_response(
            content=content,
            model=self.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def _stream(self, params: dict) -> AsyncIterator:
        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text


# ── Google (Gemini) ───────────────────────────────────────────────────────────

class GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai
        self.model_name = model
        self._client = genai.Client(api_key=api_key)

    def _build_contents(self, messages: list[dict]) -> tuple[str | None, list]:
        """Convert OpenAI-style messages to Gemini format."""
        system_text: str | None = None
        contents = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_text = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        return system_text, contents

    async def chat(self, messages: list[dict], stream: bool = False, **kwargs) -> Any:
        from google.genai import types
        system_text, contents = self._build_contents(messages)

        config_kwargs: dict = {}
        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            config_kwargs["temperature"] = kwargs.pop("temperature")
        if system_text:
            config_kwargs["system_instruction"] = system_text

        gen_config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if stream:
            return self._stream(contents, gen_config)

        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=gen_config,
        )
        content = response.text or ""
        return _make_response(content=content, model=self.model_name)

    async def _stream(self, contents, gen_config) -> AsyncIterator:
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=gen_config,
        ):
            if chunk.text:
                yield chunk.text


# ── Factory ───────────────────────────────────────────────────────────────────

def make_cloud_client(provider: str, api_key: str, model: str):
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    elif provider == "gemini":
        return GeminiClient(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown cloud provider: {provider!r}. Choose 'anthropic' or 'gemini'.")

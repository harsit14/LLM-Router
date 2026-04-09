"""
Local Model Client (LM Studio)
──────────────────────────────
LM Studio exposes an OpenAI-compatible REST API on localhost.
We use the openai SDK pointed at the local base URL.
Supports both streaming and non-streaming responses.
"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI, APIConnectionError
from typing import AsyncIterator


class LocalClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.AsyncClient(timeout=120.0),
        )

    async def is_available(self) -> bool:
        """Ping the local server to check if it's running."""
        try:
            models = await self._client.models.list()
            return len(models.data) > 0
        except (APIConnectionError, Exception):
            return False

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ):
        """
        Send a chat completion request to the local model.

        Returns:
          - If stream=False: the full completion object (OpenAI-compatible).
          - If stream=True:  an async iterator of chunk objects.
        """
        params = dict(
            model=self.model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

        if stream:
            return self._stream(params)
        else:
            return await self._client.chat.completions.create(**params)

    async def _stream(self, params: dict) -> AsyncIterator:
        # Use create(stream=True) which yields ChatCompletionChunk objects with .choices
        params = {k: v for k, v in params.items() if k != "stream"}
        async for chunk in await self._client.chat.completions.create(stream=True, **params):
            yield chunk

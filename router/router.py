"""
Core Router
───────────
Ties classifier + clients together.
  1. Classify the incoming prompt.
  2. Check local model availability (with cache so we don't ping every request).
  3. Dispatch to local or cloud.
  4. Log the decision.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.text import Text

from config import config
from router.classifier import classify, ClassificationResult
from router.local_client import LocalClient
from router.cloud_client import make_cloud_client

console = Console()

# ── Availability cache ────────────────────────────────────────────────────────
_local_available: bool | None = None
_last_ping: float = 0.0
_PING_TTL: float = 30.0  # seconds between availability checks


class LLMRouter:
    def __init__(self):
        self.local = LocalClient(
            base_url=config.LOCAL_BASE_URL,
            api_key=config.LOCAL_API_KEY,
            model=config.LOCAL_MODEL,
        )
        self.cloud = make_cloud_client(
            provider=config.CLOUD_PROVIDER,
            api_key=self._cloud_key(),
            model=self._cloud_model(),
        )

    def _cloud_key(self) -> str:
        if config.CLOUD_PROVIDER == "anthropic":
            return config.ANTHROPIC_API_KEY
        return config.GEMINI_API_KEY

    def _cloud_model(self) -> str:
        if config.CLOUD_PROVIDER == "anthropic":
            return config.ANTHROPIC_MODEL
        return config.GEMINI_MODEL

    # ── Local availability ────────────────────────────────────────────────────

    async def _local_is_up(self) -> bool:
        global _local_available, _last_ping
        now = time.monotonic()
        if _local_available is None or (now - _last_ping) > _PING_TTL:
            _local_available = await self.local.is_available()
            _last_ping = now
        return _local_available

    # ── Prompt extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(messages: list[dict]) -> str:
        """Flatten all message content into one string for classification."""
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Handle multi-modal content blocks (text only)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block["text"])
        return "\n".join(parts)

    # ── Routing decision ──────────────────────────────────────────────────────

    async def route(
        self,
        messages: list[dict],
        stream: bool = False,
        force: str | None = None,   # "local" | "cloud" | None
        **kwargs,
    ) -> Any:
        """
        Route a chat completion request.

        Args:
            messages:  OpenAI-style message list.
            stream:    Whether to stream the response.
            force:     Override routing decision ("local" or "cloud").
            **kwargs:  Extra params forwarded to the model (temperature, etc.).

        Returns:
            A response object or async iterator (if streaming).
        """
        prompt_text = self._extract_text(messages)
        classification = classify(
            prompt=prompt_text,
            threshold=config.COMPLEXITY_THRESHOLD,
            max_local_tokens=config.MAX_LOCAL_TOKENS,
        )

        # Determine actual target
        if force in ("local", "cloud"):
            target = force
            override_note = f" [forced={force}]"
        else:
            target = classification.decision
            override_note = ""

        # Fall back to cloud if local model is offline
        if target == "local" and not await self._local_is_up():
            target = "cloud"
            override_note += " [local-offline→cloud]"

        self._log(classification, target, override_note)

        if target == "local":
            return await self.local.chat(messages, stream=stream, **kwargs)
        else:
            return await self.cloud.chat(messages, stream=stream, **kwargs)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, result: ClassificationResult, target: str, note: str):
        if not config.LOG_DECISIONS:
            return

        score_color = "green" if result.score < 40 else ("yellow" if result.score < 70 else "red")
        target_color = "cyan" if target == "local" else "magenta"

        line = Text()
        line.append("⟶ ", style="bold")
        line.append(f"[{result.score:3d}/100] ", style=score_color)
        line.append(f"{target.upper()}", style=f"bold {target_color}")
        line.append(f"{note}  ", style="dim")
        line.append(f"~{result.token_estimate} tokens  ", style="dim")
        line.append(result.reason, style="dim italic")

        console.print(line)

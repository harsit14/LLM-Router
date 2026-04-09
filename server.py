"""
LLM Router — FastAPI Proxy Server
──────────────────────────────────
Exposes an OpenAI-compatible API on localhost so any tool (Continue.dev,
shell scripts, Python code) that talks to OpenAI/Claude can point here
with zero changes.

Endpoints:
  POST /v1/chat/completions   — main routing endpoint
  GET  /v1/models             — lists available models
  GET  /health                — liveness check
  GET  /classify              — inspect routing decision without calling a model
"""

from __future__ import annotations

import json
import time
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import config
from router.router import LLMRouter

app = FastAPI(
    title="LLM Router",
    description="Intelligent prompt router: local models for simple tasks, cloud for complex ones.",
    version="0.1.0",
)

router = LLMRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: Any  # str or list (multi-modal)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "auto"
    messages: list[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Routing overrides (non-standard, router-specific)
    x_force_route: Optional[str] = Field(
        default=None,
        alias="x_force_route",
        description="Force routing: 'local' or 'cloud'",
    )

    class Config:
        populate_by_name = True


class ClassifyRequest(BaseModel):
    messages: list[Message]


# ── Helper ────────────────────────────────────────────────────────────────────

def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    return [m.model_dump() for m in messages]


def _build_kwargs(req: ChatCompletionRequest) -> dict:
    kwargs = {}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    return kwargs


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Landing page — confirms the router is running."""
    local_up = await router._local_is_up()
    return {
        "service": "LLM Router",
        "status": "running",
        "local_model": config.LOCAL_MODEL,
        "local_available": local_up,
        "cloud": f"{config.CLOUD_PROVIDER}/{router._cloud_model()}",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "health": "GET /health",
            "classify": "POST /classify",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
async def health():
    local_up = await router._local_is_up()
    return {
        "status": "ok",
        "local_model": {"url": config.LOCAL_BASE_URL, "model": config.LOCAL_MODEL, "available": local_up},
        "cloud": {"provider": config.CLOUD_PROVIDER, "model": router._cloud_model()},
        "threshold": config.COMPLEXITY_THRESHOLD,
    }


@app.get("/v1/models")
async def list_models():
    """Return a fake models list so OpenAI SDK clients don't complain."""
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "created": int(time.time()), "owned_by": "llm-router"},
            {"id": config.LOCAL_MODEL, "object": "model", "created": int(time.time()), "owned_by": "local"},
            {"id": router._cloud_model(), "object": "model", "created": int(time.time()), "owned_by": config.CLOUD_PROVIDER},
        ],
    }


@app.post("/classify")
async def classify_only(req: ClassifyRequest):
    """
    Inspect the routing decision for a set of messages without actually
    calling any model. Useful for debugging or understanding the classifier.
    """
    from router.classifier import classify as do_classify
    messages = _messages_to_dicts(req.messages)
    text = router._extract_text(messages)
    result = do_classify(text, config.COMPLEXITY_THRESHOLD, config.MAX_LOCAL_TOKENS)
    local_up = await router._local_is_up()
    effective = result.decision if (result.decision == "cloud" or local_up) else "cloud (local offline)"
    return {
        "score": result.score,
        "threshold": config.COMPLEXITY_THRESHOLD,
        "raw_decision": result.decision,
        "effective_decision": effective,
        "reason": result.reason,
        "token_estimate": result.token_estimate,
        "local_available": local_up,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    messages = _messages_to_dicts(req.messages)
    kwargs = _build_kwargs(req)

    # Allow forcing route via custom header OR request body field
    force = req.x_force_route or request.headers.get("X-Force-Route")

    try:
        if req.stream:
            return await _handle_streaming(messages, kwargs, force)
        else:
            return await _handle_standard(messages, kwargs, force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _handle_standard(messages: list[dict], kwargs: dict, force: str | None) -> JSONResponse:
    response = await router.route(messages, stream=False, force=force, **kwargs)

    # If it's already a dict (our cloud shim), return directly
    if isinstance(response, dict):
        return JSONResponse(content=response)

    # OpenAI SDK object — serialise
    return JSONResponse(content=response.model_dump())


async def _handle_streaming(messages: list[dict], kwargs: dict, force: str | None) -> StreamingResponse:
    async def generator():
        stream = await router.route(messages, stream=True, force=force, **kwargs)
        async for chunk in stream:
            if isinstance(chunk, str):
                # Gemini / Anthropic text chunks — wrap in SSE OpenAI format
                data = {
                    "id": f"router-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
            else:
                # OpenAI SDK chunk object
                data = chunk.model_dump()

            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    c = Console()
    c.print(Panel.fit(
        Text.from_markup(
            f"[bold cyan]LLM Router[/bold cyan]\n"
            f"Local  → [green]{config.LOCAL_MODEL}[/green] @ {config.LOCAL_BASE_URL}\n"
            f"Cloud  → [magenta]{config.CLOUD_PROVIDER}/{router._cloud_model()}[/magenta]\n"
            f"Threshold [bold]{config.COMPLEXITY_THRESHOLD}/100[/bold]  |  "
            f"Max local tokens [bold]{config.MAX_LOCAL_TOKENS}[/bold]\n\n"
            f"Proxy URL: [underline]http://{config.SERVER_HOST}:{config.SERVER_PORT}/v1[/underline]"
        ),
        title="Starting",
        border_style="cyan",
    ))

    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload="--reload" in sys.argv,
        log_level="warning",  # suppress uvicorn noise; router logs its own decisions
    )

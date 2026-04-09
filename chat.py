"""
Interactive CLI chat for LLM Router
────────────────────────────────────
Streams responses directly — no server needed.

Usage:
  python chat.py                  # auto-route (default)
  python chat.py --local          # force local model only
  python chat.py --cloud          # force cloud model only
  python chat.py --model gemini   # override cloud model

Commands during chat:
  /exit or /quit  — quit
  /clear          — clear conversation history
  /route          — show where the last message was routed
  /info           — show current config
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt

from config import config
from router.router import LLMRouter
from router.classifier import classify

console = Console()
router = LLMRouter()


def parse_args() -> dict:
    args = sys.argv[1:]
    force = None
    if "--local" in args:
        force = "local"
    elif "--cloud" in args:
        force = "cloud"
    return {"force": force}


def print_header(force: str | None):
    local_label = f"[green]{config.LOCAL_MODEL}[/green]"
    cloud_label = f"[magenta]{config.CLOUD_PROVIDER}/{router._cloud_model()}[/magenta]"

    if force == "local":
        mode = f"Forced → {local_label}"
    elif force == "cloud":
        mode = f"Forced → {cloud_label}"
    else:
        mode = f"Auto-route  local={local_label}  cloud={cloud_label}  threshold={config.COMPLEXITY_THRESHOLD}/100"

    console.print(Panel.fit(
        Text.from_markup(f"[bold cyan]LLM Router Chat[/bold cyan]\n{mode}\n\n[dim]/clear  /route  /info  /exit[/dim]"),
        border_style="cyan",
    ))


async def stream_response(messages: list[dict], force: str | None) -> tuple[str, str]:
    """Stream a response and return (full_text, target)."""
    # Classify first so we can show target before streaming
    prompt_text = router._extract_text(messages)
    result = classify(prompt_text, config.COMPLEXITY_THRESHOLD, config.MAX_LOCAL_TOKENS)

    # Determine actual target (mirrors router logic)
    if force in ("local", "cloud"):
        target = force
    else:
        target = result.decision

    local_up = await router._local_is_up()
    if target == "local" and not local_up:
        target = "cloud"
        console.print("[yellow]⚠ Local model offline — falling back to cloud[/yellow]")

    # Show routing decision
    score_color = "green" if result.score < 40 else ("yellow" if result.score < 70 else "red")
    target_color = "cyan" if target == "local" else "magenta"
    console.print(
        f"[dim]⟶ [{score_color}]{result.score}/100[/{score_color}]  "
        f"[{target_color}]{target.upper()}[/{target_color}]  {result.reason}[/dim]"
    )

    # Stream the actual response
    console.print()
    full_text = ""

    try:
        stream = await router.route(messages, stream=True, force=force)
        async for chunk in stream:
            if isinstance(chunk, str):
                text = chunk
            else:
                # OpenAI SDK chunk
                delta = chunk.choices[0].delta if chunk.choices else None
                text = (delta.content or "") if delta else ""

            if text:
                full_text += text
                console.print(text, end="", highlight=False)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")

    console.print("\n")
    return full_text, target


async def chat_loop(force: str | None):
    print_header(force)

    messages: list[dict] = []
    last_target: str = "?"

    while True:
        try:
            user_input = Prompt.ask("\n[bold]You[/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        if user_input.lower() == "/route":
            console.print(f"[dim]Last message routed to: [bold]{last_target}[/bold][/dim]")
            continue

        if user_input.lower() == "/info":
            local_up = await router._local_is_up()
            console.print(
                f"[dim]Local: {config.LOCAL_MODEL} @ {config.LOCAL_BASE_URL}  "
                f"({'[green]online[/green]' if local_up else '[red]offline[/red]'})\n"
                f"Cloud: {config.CLOUD_PROVIDER}/{router._cloud_model()}\n"
                f"Threshold: {config.COMPLEXITY_THRESHOLD}/100  Max local tokens: {config.MAX_LOCAL_TOKENS}[/dim]"
            )
            continue

        # Normal message
        messages.append({"role": "user", "content": user_input})

        console.print("\n[bold]Assistant[/bold] ", end="")
        response_text, last_target = await stream_response(messages, force)

        if response_text:
            messages.append({"role": "assistant", "content": response_text})


def main():
    opts = parse_args()
    asyncio.run(chat_loop(opts["force"]))


if __name__ == "__main__":
    main()

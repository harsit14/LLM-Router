"""
Quick test script — runs without a server.
Tests the classifier and (optionally) live model calls.

Usage:
  python test_router.py              # classifier only
  python test_router.py --live       # classifier + real model calls
"""

import asyncio
import sys
from rich.console import Console
from rich.table import Table

console = Console()


# ── Classifier tests ──────────────────────────────────────────────────────────

TEST_PROMPTS = [
    # (prompt, expected_decision)
    ("Summarize this email: ...", "local"),
    ("Fix grammar: 'She dont like it'", "local"),
    ("Translate 'hello world' to French", "local"),
    ("What is photosynthesis?", "local"),
    ("Rewrite this sentence in a formal tone.", "local"),
    ("Extract all dates from this text: ...", "local"),
    ("Design a distributed caching system for a social media platform with 50M DAU.", "cloud"),
    ("Implement a binary search tree with insertion, deletion, and balancing.", "cloud"),
    ("Compare and contrast microservices vs monolithic architecture. Analyze trade-offs.", "cloud"),
    ("Debug this React component — it re-renders infinitely.", "cloud"),
    ("Prove that the square root of 2 is irrational.", "cloud"),
    ("Write a comprehensive research summary on transformer attention mechanisms.", "cloud"),
]


def run_classifier_tests():
    from router.classifier import classify
    from config import config

    table = Table(title="Classifier Test Results", show_lines=True)
    table.add_column("Prompt", style="white", max_width=55)
    table.add_column("Score", justify="right")
    table.add_column("Decision", justify="center")
    table.add_column("Expected", justify="center")
    table.add_column("Pass", justify="center")
    table.add_column("Reason", style="dim", max_width=40)

    passed = 0
    for prompt, expected in TEST_PROMPTS:
        result = classify(prompt, config.COMPLEXITY_THRESHOLD, config.MAX_LOCAL_TOKENS)
        ok = result.decision == expected
        if ok:
            passed += 1
        score_color = "green" if result.score < 40 else ("yellow" if result.score < 70 else "red")
        table.add_row(
            prompt[:55],
            f"[{score_color}]{result.score}[/{score_color}]",
            f"[cyan]{result.decision}[/cyan]" if result.decision == "local" else f"[magenta]{result.decision}[/magenta]",
            expected,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
            result.reason,
        )

    console.print(table)
    console.print(f"\n[bold]Classifier accuracy: {passed}/{len(TEST_PROMPTS)} ({100*passed//len(TEST_PROMPTS)}%)[/bold]")


# ── Live model test (optional) ────────────────────────────────────────────────

async def run_live_test():
    from router.router import LLMRouter
    r = LLMRouter()

    simple_msg = [{"role": "user", "content": "Summarize this in one sentence: The quick brown fox jumps over the lazy dog."}]
    complex_msg = [{"role": "user", "content": "Compare microservices vs monolithic architecture. What are the trade-offs for a startup?"}]

    console.rule("[bold]Live Routing Test[/bold]")

    for label, messages in [("Simple prompt", simple_msg), ("Complex prompt", complex_msg)]:
        console.print(f"\n[bold yellow]{label}[/bold yellow]: {messages[0]['content'][:60]}...")
        try:
            response = await r.route(messages)
            if isinstance(response, dict):
                content = response["choices"][0]["message"]["content"]
            else:
                content = response.choices[0].message.content
            console.print(f"[dim]Response:[/dim] {content[:200]}{'...' if len(content) > 200 else ''}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold cyan]LLM Router — Tests[/bold cyan]")
    run_classifier_tests()

    if "--live" in sys.argv:
        asyncio.run(run_live_test())
    else:
        console.print("\n[dim]Run with --live to also test real model calls.[/dim]")

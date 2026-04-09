"""
Complexity Classifier
─────────────────────
Scores an incoming prompt 0–100 (higher = more complex).
Uses a fast, zero-dependency heuristic pipeline so routing adds <1ms latency.

Score breakdown:
  - Token count          →  0–25 pts
  - Task-type keywords   →  0–30 pts
  - Structural signals   →  0–25 pts
  - Reasoning cues       →  0–20 pts

Total ≥ threshold (default 50) → route to cloud.
"""

import re
from dataclasses import dataclass


# ── Keyword tables ────────────────────────────────────────────────────────────

# Tasks that are almost always simple enough for a local model
SIMPLE_KEYWORDS = {
    # text ops
    "summarize", "summary", "tldr", "rewrite", "rephrase", "paraphrase",
    "fix grammar", "grammar", "spell check", "spellcheck", "translate",
    "format", "bullet points", "list the", "extract",
    # basic Q&A
    "what is", "what are", "define", "definition", "meaning of",
    "how do i", "how to", "steps to", "give me a recipe",
    # tone / style
    "make it shorter", "make it longer", "make it formal", "make it casual",
    "email subject", "subject line",
    # code (simple)
    "rename", "add a comment", "explain this function", "what does this code do",
}

# Tasks that hint at cloud-level complexity
COMPLEX_KEYWORDS = {
    # reasoning / analysis
    "analyze", "analysis", "compare", "contrast", "evaluate", "critique",
    "pros and cons", "trade-offs", "tradeoffs", "implications",
    # planning / architecture
    "design", "architect", "system design", "plan", "roadmap", "strategy",
    "best approach", "recommend", "advise",
    # research / synthesis
    "research", "literature", "synthesize", "survey",
    "explain in depth", "deep dive", "comprehensive",
    # coding (complex)
    "implement", "build", "create a", "write a", "develop", "refactor",
    "debug", "fix the bug", "optimize", "performance",
    # math / logic
    "prove", "proof", "calculate", "derive", "solve", "equation",
    "algorithm", "complexity", "big o",
    # multi-step
    "step by step", "walk me through", "in detail",
}

# Phrases that signal multi-turn or chained reasoning
REASONING_CUES = [
    r"\bfirst\b.{0,80}\bthen\b",
    r"\bif\b.{0,80}\bthen\b.{0,80}\belse\b",
    r"\bwhy\b.{0,40}\bhow\b",
    r"\bcompare\b.{0,60}\band\b",
    r"\band also\b",
    r"\bmoreover\b",
    r"\bhowever\b",
    r"\bon the other hand\b",
    r"\bgiven that\b",
    r"\btaking into account\b",
]

# Structural signals that inflate complexity
STRUCTURAL_SIGNALS = {
    "code_block": r"```",
    "numbered_list": r"^\s*\d+\.",
    "multiple_questions": r"\?.*\?",
    "long_url": r"https?://\S{40,}",
    "json_xml": r"[{<]\s*\"?\w+\"?\s*[:{<]",
    "table": r"\|.+\|.+\|",
}


# ── Approximate token count (no tiktoken needed for heuristics) ───────────────

def _approx_tokens(text: str) -> int:
    """GPT-style rough estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── Scorer ────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    score: int              # 0–100
    decision: str           # "local" | "cloud"
    reason: str             # human-readable explanation
    token_estimate: int


def classify(prompt: str, threshold: int = 50, max_local_tokens: int = 3000) -> ClassificationResult:
    """
    Classify a prompt and return a routing decision.

    Args:
        prompt:           The full prompt text (all messages concatenated).
        threshold:        Score at or above which we route to cloud (0–100).
        max_local_tokens: Hard cap — always cloud if prompt exceeds this.

    Returns:
        ClassificationResult with score, decision, and reasoning.
    """
    text = prompt.lower().strip()
    token_est = _approx_tokens(prompt)
    reasons: list[str] = []
    score = 0

    # ── 1. Hard override: token count ────────────────────────────────────────
    if token_est > max_local_tokens:
        return ClassificationResult(
            score=100,
            decision="cloud",
            reason=f"Prompt too long for local model ({token_est} tokens > {max_local_tokens} limit)",
            token_estimate=token_est,
        )

    # ── 2. Token count contribution (0–25 pts) ───────────────────────────────
    if token_est > 1500:
        score += 25
        reasons.append(f"long prompt ({token_est} tokens)")
    elif token_est > 700:
        score += 15
        reasons.append(f"medium-length prompt ({token_est} tokens)")
    elif token_est > 300:
        score += 7

    # ── 3. Task-type keywords (0–50 pts) ─────────────────────────────────────
    complex_hits = [kw for kw in COMPLEX_KEYWORDS if kw in text]
    simple_hits  = [kw for kw in SIMPLE_KEYWORDS  if kw in text]

    # Each complex keyword is worth enough to push past threshold on its own.
    # If we hit the cap, don't let simple keywords drag us back below threshold.
    raw_complex = min(threshold, len(complex_hits) * (threshold + 5))
    if raw_complex >= threshold:
        keyword_score = raw_complex  # already at cap — simple keywords can't cancel
    else:
        simple_penalty = min(15, len(simple_hits) * 5)
        keyword_score = max(0, raw_complex - simple_penalty)
    score += keyword_score

    if complex_hits:
        reasons.append(f"complex keywords: {', '.join(complex_hits[:3])}")
    if simple_hits:
        reasons.append(f"simple keywords: {', '.join(simple_hits[:3])}")

    # ── 4. Structural signals (0–25 pts) ─────────────────────────────────────
    struct_hits: list[str] = []
    for name, pattern in STRUCTURAL_SIGNALS.items():
        if re.search(pattern, prompt, re.MULTILINE | re.IGNORECASE):
            struct_hits.append(name)

    struct_score = min(25, len(struct_hits) * 8)
    score += struct_score
    if struct_hits:
        reasons.append(f"structural signals: {', '.join(struct_hits)}")

    # ── 5. Reasoning / multi-step cues (0–20 pts) ────────────────────────────
    reasoning_hits = [p for p in REASONING_CUES if re.search(p, text)]
    reasoning_score = min(20, len(reasoning_hits) * 7)
    score += reasoning_score
    if reasoning_hits:
        reasons.append("multi-step / reasoning language detected")

    # ── 6. Question complexity ────────────────────────────────────────────────
    question_count = text.count("?")
    if question_count >= 3:
        score += 10
        reasons.append(f"{question_count} questions in prompt")
    elif question_count == 2:
        score += 5

    # ── 7. Cap and decide ────────────────────────────────────────────────────
    score = min(100, max(0, score))
    decision = "cloud" if score >= threshold else "local"
    reason_str = "; ".join(reasons) if reasons else "low complexity heuristics"

    return ClassificationResult(
        score=score,
        decision=decision,
        reason=reason_str,
        token_estimate=token_est,
    )

"""
Central configuration for LLM Router.
All values can be overridden via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Local model (LM Studio) ───────────────────────────────────────────────
    LOCAL_BASE_URL: str = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
    LOCAL_API_KEY: str = os.getenv("LOCAL_API_KEY", "lm-studio")  # LM Studio ignores key
    LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "qwen2.5-32b-instruct")

    # ── Cloud model ───────────────────────────────────────────────────────────
    CLOUD_PROVIDER: str = os.getenv("CLOUD_PROVIDER", "anthropic")  # "anthropic" | "gemini"

    # Anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Routing thresholds ────────────────────────────────────────────────────
    # Complexity score 0–100. Requests scoring below this go local.
    COMPLEXITY_THRESHOLD: int = int(os.getenv("COMPLEXITY_THRESHOLD", "50"))

    # Hard token limit: prompts over this always go to cloud regardless of score
    MAX_LOCAL_TOKENS: int = int(os.getenv("MAX_LOCAL_TOKENS", "3000"))

    # ── Proxy server ──────────────────────────────────────────────────────────
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8080"))

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_DECISIONS: bool = os.getenv("LOG_DECISIONS", "true").lower() == "true"


config = Config()

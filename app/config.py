"""Runtime configuration, read once from the environment.

Provider-agnostic: the workflow runs on OpenAI or Anthropic depending on
``LLM_PROVIDER``. Nothing in the agent code knows which is in use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Model ------------------------------------------------------------
    provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai").lower())
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    #: 0 for reproducibility. Two identical applications must decide identically
    #: -- for lending that is a legal requirement, not a preference.
    temperature: float = 0.0

    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "").strip())
    #: Optional proxy base (the JHU course routes OpenAI through its own gateway).
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "").strip())
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "").strip())

    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Retrieval --------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 200))
    retrieval_k: int = field(default_factory=lambda: _int("RETRIEVAL_K", 6))

    # --- Spend protection for the public demo -----------------------------
    #: A single run is ~6 model calls. Without these a shared link is an
    #: unbounded invoice.
    rate_limit_per_hour: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_HOUR", 5))
    daily_run_cap: int = field(default_factory=lambda: _int("DAILY_RUN_CAP", 200))

    policy_pdf: Path = DATA_DIR / "underwriting_policies.pdf"
    test_cases: Path = DATA_DIR / "mortgage_test_cases.json"

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return "claude-sonnet-4-5" if self.provider == "anthropic" else "gpt-4o-mini"

    def validate(self) -> list[str]:
        """Return a list of configuration problems, empty when healthy."""
        problems: list[str] = []
        if self.provider == "anthropic":
            if not self.anthropic_api_key:
                problems.append("ANTHROPIC_API_KEY is not set")
            # Embeddings always go through OpenAI; Anthropic serves chat only.
            if not self.openai_api_key:
                problems.append("OPENAI_API_KEY is required for embeddings even on Anthropic")
        elif not self.openai_api_key:
            problems.append("OPENAI_API_KEY is not set")
        if not self.policy_pdf.exists():
            problems.append(f"policy PDF missing at {self.policy_pdf}")
        if not self.test_cases.exists():
            problems.append(f"test cases missing at {self.test_cases}")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

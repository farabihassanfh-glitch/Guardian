"""Runtime configuration, read once from the environment.

Provider-agnostic by design. The default deployment runs entirely on a single
Anthropic API key: Claude for the agents, local ONNX embeddings for retrieval.
OpenAI remains supported for anyone who has a key for it.
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
    # --- Chat model -------------------------------------------------------
    provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "anthropic").lower()
    )
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "").strip())

    #: Reasoning depth. Claude only; ignored on OpenAI.
    #: low | medium | high | xhigh | max
    effort: str = field(default_factory=lambda: os.environ.get("LLM_EFFORT", "medium").lower())

    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "").strip()
    )
    #: Optional proxy base for OpenAI-compatible gateways.
    openai_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "").strip()
    )

    # --- Embeddings -------------------------------------------------------
    #: "local" runs a small ONNX model in-process — no API key, no second
    #: vendor. Anthropic serves no embeddings endpoint, so this is what makes
    #: a Claude-only deployment possible.
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_PROVIDER", "local").lower()
    )
    embedding_model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", ""))

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
        return "gpt-4o-mini" if self.provider == "openai" else "claude-opus-5"

    @property
    def resolved_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        if self.embedding_provider == "openai":
            return "text-embedding-3-small"
        # 384-dim, ~130MB ONNX. Downloaded once and cached on first use.
        return "BAAI/bge-small-en-v1.5"

    def validate(self) -> list[str]:
        """Return a list of configuration problems, empty when healthy."""
        problems: list[str] = []

        if self.provider == "anthropic":
            if not self.anthropic_api_key:
                problems.append("ANTHROPIC_API_KEY is not set")
        elif self.provider == "openai":
            if not self.openai_api_key:
                problems.append("OPENAI_API_KEY is not set")
        else:
            problems.append(f"LLM_PROVIDER must be 'anthropic' or 'openai', got '{self.provider}'")

        if self.embedding_provider == "openai" and not self.openai_api_key:
            problems.append("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")

        if not self.policy_pdf.exists():
            problems.append(f"policy PDF missing at {self.policy_pdf}")
        if not self.test_cases.exists():
            problems.append(f"test cases missing at {self.test_cases}")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

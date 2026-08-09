"""Model access, isolated behind two functions.

Agents never construct a client. They ask for ``get_llm()`` and get whatever
the environment is configured for, so switching provider is an env var rather
than a sweep through six agent modules.

Two provider differences are handled here so nothing downstream has to know:

**Sampling parameters.** ``temperature`` is *rejected* on Claude Opus 5 and
later — the request returns a 400, it isn't merely ignored. Determinism on
Claude comes from prompt design and low effort, not a sampling knob. The
OpenAI path still sets ``temperature=0``.

**Embeddings.** Anthropic serves no embeddings endpoint. Rather than require a
second vendor's key just to build a vector index over a 14-page PDF, the
default runs a small ONNX model in-process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import get_settings

log = logging.getLogger(__name__)


def response_text(response: Any) -> str:
    """Flatten a chat response to plain text.

    Providers disagree about the shape here, and the difference is silent until
    something downstream calls a string method. OpenAI returns ``content`` as a
    ``str``. Anthropic returns a **list of content blocks** whenever thinking is
    on — which is the default on Claude Opus 5 — so ``content`` looks like
    ``[{"type": "thinking", ...}, {"type": "text", "text": "..."}]``.

    Thinking blocks are dropped deliberately: reasoning is not part of the
    credit memo, and it should never reach the bias scanner or the audit trail.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p).strip()

    return str(content)


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Return the configured chat model."""
    s = get_settings()

    if s.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": s.resolved_model,
            "temperature": 0,
            "api_key": s.openai_api_key,
        }
        if s.openai_base_url:
            kwargs["base_url"] = s.openai_base_url
        return ChatOpenAI(**kwargs)

    from langchain_anthropic import ChatAnthropic

    # No `temperature` — Claude Opus 5 rejects sampling parameters outright.
    # `effort` bounds reasoning depth and is the cost/latency lever in its place.
    return ChatAnthropic(
        model=s.resolved_model,
        api_key=s.anthropic_api_key,
        max_tokens=8192,
        output_config={"effort": s.effort},
    )


class LocalEmbeddings(Embeddings):
    """LangChain embeddings backed by a local ONNX model.

    Thirty lines instead of a dependency on a sunset package — and it removes
    the last reason this service would need a second vendor's API key. The
    model is downloaded once (~130 MB) and cached on disk; after that,
    retrieval involves no network at all.
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        log.info("Loading local embedding model %s", model_name)
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the configured embedding model.

    ``local`` is the default. The policy manual is 14 pages, so model size is
    not the constraint on retrieval quality here — chunking is.
    """
    s = get_settings()

    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": s.resolved_embedding_model,
            "api_key": s.openai_api_key,
        }
        if s.openai_base_url:
            kwargs["base_url"] = s.openai_base_url
        return OpenAIEmbeddings(**kwargs)

    return LocalEmbeddings(s.resolved_embedding_model)

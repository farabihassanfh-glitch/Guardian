"""Model access, isolated behind one function.

Agents never construct a client. They ask for ``get_llm()`` and get whatever the
environment is configured for, so switching provider is an env var rather than a
sweep through six agent modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Return the configured chat model."""
    s = get_settings()

    if s.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=s.resolved_model,
            temperature=s.temperature,
            api_key=s.anthropic_api_key,
            max_tokens=4096,
        )

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": s.resolved_model,
        "temperature": s.temperature,
        "api_key": s.openai_api_key,
    }
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """Return the embedding model.

    Always OpenAI: Anthropic does not serve an embeddings endpoint, so an
    Anthropic deployment still needs an OpenAI key for retrieval.
    """
    from langchain_openai import OpenAIEmbeddings

    s = get_settings()
    kwargs: dict[str, Any] = {"model": s.embedding_model, "api_key": s.openai_api_key}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return OpenAIEmbeddings(**kwargs)

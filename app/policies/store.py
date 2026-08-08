"""Policy retrieval (RAG) over the underwriting manual.

Built once at process start, held in memory. Rebuilding per request would
re-embed the whole manual on every visit -- roughly 45 embedding calls and 30
seconds of latency for a document that has not changed.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from app.config import get_settings
from app.llm import get_embeddings

log = logging.getLogger(__name__)

_store: Any | None = None
_lock = threading.Lock()

#: Matches a section heading such as "2.3 Self-Employment Income" at the start
#: of a chunk, used to group overlapping chunks back into coherent sections.
_SECTION_RE = re.compile(r"^(\d+\.\d+\s+[A-Z][A-Za-z ,&-]+)")


def build_policy_store() -> Any:
    """Load the policy PDF, chunk it, embed it and return a vector store."""
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    s = get_settings()
    log.info("Loading policy manual from %s", s.policy_pdf)

    documents = PyPDFLoader(str(s.policy_pdf)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        # Prefer paragraph breaks so a rule is not severed from its exception.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info("Split %d pages into %d chunks", len(documents), len(chunks))

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="underwriting_policies",
    )
    log.info("Policy vector store ready")
    return store


def get_policy_store() -> Any:
    """Return the process-wide store, building it on first use."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_policy_store()
    return _store


def retrieve_policies(query: str, k: int | None = None) -> str:
    """Return the policy text most relevant to ``query``, deduplicated by section.

    Chunks overlap by design, and several results usually come from the same
    section, so raw concatenation feeds the model the same paragraph three or
    four times. Grouping by heading keeps the context small and readable.
    """
    settings = get_settings()
    docs = get_policy_store().similarity_search(query, k=k or settings.retrieval_k)

    sections: dict[str, str] = {}
    for doc in docs:
        text = doc.page_content.strip()
        match = _SECTION_RE.match(text)
        key = match.group(1) if match else f"_unlabelled_{len(sections)}"
        if key not in sections:
            sections[key] = text
        elif text not in sections[key]:
            sections[key] += "\n" + text

    return "\n\n".join(sections.values())

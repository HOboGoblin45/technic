"""Codex-facing context layer for embeddings and retrieval."""

from __future__ import annotations

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False
    faiss = None
    SentenceTransformer = None


def build_context_vector(docs):
    """Build a FAISS index from a list of documents using sentence embeddings."""
    if not HAVE_DEPS:
        raise ImportError("faiss and sentence-transformers are required for context indexing.")
    if not docs:
        raise ValueError("docs must be non-empty")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(docs)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)
    return index


__all__ = ["build_context_vector"]

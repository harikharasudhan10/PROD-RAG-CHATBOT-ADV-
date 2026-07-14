"""
embedding.py
============
Embedding generation, FAISS index management, MMR retrieval, and
CrossEncoder reranking for the ML-Bot RAG pipeline.

Public API:
    load_embedding_models()        – load all three sentence-encoders
    generate_embeddings_for_all()  – embed every chunk_type × model combo
    save_embeddings() / load_embeddings()
    save_chunk_sets() / load_chunk_sets()
    build_and_save_indexes()       – build + persist FAISS indexes
    load_faiss_index()             – load one index from disk
    mmr_search()                   – diversity-aware retrieval
    load_reranker()                – load CrossEncoder
    rerank_documents()             – rerank retrieved chunks

Quick start:
    from embedding import (
        load_embedding_models, generate_embeddings_for_all,
        build_and_save_indexes, load_faiss_index,
        mmr_search, load_reranker, rerank_documents,
    )
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ChunkDict      = dict[str, Any]
EmbeddingStore = dict[str, dict[str, dict[str, Any]]]
# EmbeddingStore schema:
#   { chunk_type: { model_name: { "embeddings": np.ndarray, "texts": list[str] } } }


# ---------------------------------------------------------------------------
# 1.  Embedding-model registry
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: dict[str, str] = {
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
    "bge":    "BAAI/bge-small-en",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def load_embedding_models(
    device: str = "cuda",
    model_keys: list[str] | None = None,
) -> dict[str, SentenceTransformer]:
    """
    Load sentence-transformer models defined in :data:`EMBEDDING_MODELS`.

    Args:
        device:     ``"cuda"`` or ``"cpu"``.
        model_keys: Subset of keys to load.  ``None`` loads all three.

    Returns:
        Dict mapping short key (e.g. ``"bge"``) → ``SentenceTransformer``.
    """
    keys = model_keys or list(EMBEDDING_MODELS)
    return {k: SentenceTransformer(EMBEDDING_MODELS[k], device=device) for k in keys}


# ---------------------------------------------------------------------------
# 2.  Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings_for_all(
    chunk_sets: dict[str, list[ChunkDict]],
    loaded_models: dict[str, SentenceTransformer],
    batch_size: int = 32,
) -> EmbeddingStore:
    """
    Encode every (chunk_type × model) combination with L2-normalised embeddings.

    Args:
        chunk_sets:    ``{"recursive": [...], "hybrid": [...], ...}``.
        loaded_models: Output of :func:`load_embedding_models`.
        batch_size:    Encoding batch size (tune for VRAM).

    Returns:
        Nested dict: ``{chunk_type: {model_name: {"embeddings": ndarray, "texts": list}}}``.
    """
    results: EmbeddingStore = {}

    for chunk_type, chunks in chunk_sets.items():
        texts = [c["text"] for c in chunks]
        results[chunk_type] = {}

        for model_name, model in loaded_models.items():
            print(f"  Embedding  [{chunk_type}] × [{model_name}]  ({len(texts)} chunks) …")
            vecs = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            results[chunk_type][model_name] = {
                "embeddings": np.array(vecs, dtype=np.float32),
                "texts":      texts,
            }

    return results


# ---------------------------------------------------------------------------
# 3.  Persistence helpers
# ---------------------------------------------------------------------------

def save_embeddings(
    store: EmbeddingStore,
    path: str | Path = "all_embeddings_results.pkl",
) -> None:
    """Serialise the full embedding store to disk with pickle."""
    with open(path, "wb") as fh:
        pickle.dump(store, fh)
    print(f"  Embeddings saved → {path}")


def load_embeddings(path: str | Path = "all_embeddings_results.pkl") -> EmbeddingStore:
    """Deserialise an embedding store previously saved by :func:`save_embeddings`."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_chunk_sets(
    chunk_sets: dict[str, list[ChunkDict]],
    path: str | Path = "chunk_sets.pkl",
) -> None:
    """Serialise chunk sets to disk with pickle."""
    with open(path, "wb") as fh:
        pickle.dump(chunk_sets, fh)
    print(f"  Chunk sets saved → {path}")


def load_chunk_sets(path: str | Path = "chunk_sets.pkl") -> dict[str, list[ChunkDict]]:
    """Deserialise chunk sets previously saved by :func:`save_chunk_sets`."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# 4.  FAISS index management
# ---------------------------------------------------------------------------

def _index_path(chunk_type: str, model_name: str) -> str:
    return f"{chunk_type}_{model_name}_faiss_index.bin"


def build_and_save_indexes(
    store: EmbeddingStore,
    chunk_types: list[str] | None = None,
    model_names: list[str] | None = None,
) -> dict[str, faiss.IndexFlatL2]:
    """
    Build one ``IndexFlatL2`` FAISS index per (chunk_type × model) combination
    and write each to ``{chunk_type}_{model_name}_faiss_index.bin``.

    Args:
        store:       Output of :func:`generate_embeddings_for_all`.
        chunk_types: Subset of chunk types to process.  ``None`` = all.
        model_names: Subset of model keys to process.  ``None`` = all.

    Returns:
        Dict mapping ``"{chunk_type}_{model_name}"`` → populated index.
    """
    chunk_types = chunk_types or list(store)
    indexes: dict[str, faiss.IndexFlatL2] = {}

    for ct in chunk_types:
        models = model_names or list(store[ct])

        for mn in models:
            vecs = store[ct][mn]["embeddings"].astype(np.float32)
            dim  = vecs.shape[1]

            index = faiss.IndexFlatL2(dim)
            index.add(vecs)

            path = _index_path(ct, mn)
            faiss.write_index(index, path)
            print(f"  FAISS [{ct}] × [{mn}]  {index.ntotal} vectors (dim {dim}) → {path}")

            indexes[f"{ct}_{mn}"] = index

    return indexes


def load_faiss_index(chunk_type: str, model_name: str) -> faiss.IndexFlatL2:
    """
    Load a previously saved FAISS index from disk.

    Args:
        chunk_type:  e.g. ``"hybrid"`` or ``"recursive"``.
        model_name:  e.g. ``"minilm"``, ``"bge"``, or ``"mpnet"``.

    Returns:
        Populated ``faiss.IndexFlatL2`` instance.

    Raises:
        FileNotFoundError: If the ``.bin`` file does not exist.
    """
    path = _index_path(chunk_type, model_name)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"FAISS index not found: {path}\n"
            "Run build_and_save_indexes() first (via build_index.py)."
        )
    return faiss.read_index(path)


# ---------------------------------------------------------------------------
# 5.  MMR retrieval
# ---------------------------------------------------------------------------

def mmr_search(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    document_texts: list[str],
    faiss_index: faiss.IndexFlatL2,
    k: int = 5,
    lambda_mult: float = 0.7,
    fetch_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Maximum Marginal Relevance (MMR) retrieval.

    Fetches *fetch_k* nearest neighbours from the FAISS index, then
    greedily selects *k* results that balance relevance to the query
    against diversity among already-selected results.

    Args:
        query_embedding:    Float32 array of shape ``(1, dim)``.
        document_embeddings: Float32 array of shape ``(n, dim)``.
        document_texts:     Text strings aligned with *document_embeddings*.
        faiss_index:        Populated FAISS index for candidate retrieval.
        k:                  Number of final diverse results to return.
        lambda_mult:        ``1.0`` = max relevance, ``0.0`` = max diversity.
        fetch_k:            How many candidates to pull from FAISS initially.

    Returns:
        List of dicts with keys ``text``, ``distance``, ``index``.
        Results are ordered by MMR selection (highest-scoring first).
    """
    fetch_k = min(fetch_k, len(document_embeddings))

    raw_dists, raw_idxs = faiss_index.search(query_embedding, fetch_k)

    # Filter out any out-of-range indices (can occur when fetch_k > index size)
    valid = [i for i in raw_idxs[0] if i < len(document_embeddings)]
    if not valid:
        return []

    cand_embs  = document_embeddings[valid]
    cand_texts = [document_texts[i] for i in valid]
    cand_dists = raw_dists[0][: len(valid)]

    # L2 distance → similarity proxy  (lower distance ≈ higher similarity)
    query_sims = 1.0 - cand_dists

    selected:   list[int] = []
    unselected: list[int] = list(range(len(cand_embs)))

    while len(selected) < k and unselected:
        scores: list[float] = []

        for ci in unselected:
            relevance = float(query_sims[ci])
            diversity = 0.0

            if selected:
                sel_embs   = cand_embs[selected]
                sims_to_sel = cosine_similarity([cand_embs[ci]], sel_embs)[0]
                diversity  = float(np.max(sims_to_sel))

            scores.append(lambda_mult * relevance - (1 - lambda_mult) * diversity)

        best_local  = int(np.argmax(scores))
        best_global = unselected[best_local]
        selected.append(best_global)
        unselected.pop(best_local)

    return [
        {
            "text":     cand_texts[i],
            "distance": float(cand_dists[i]),
            "index":    valid[i],
        }
        for i in selected
    ]


# ---------------------------------------------------------------------------
# 6.  CrossEncoder reranking
# ---------------------------------------------------------------------------

def load_reranker(
    model_name: str = "BAAI/bge-reranker-base",
    device: str = "cuda",
) -> CrossEncoder:
    """
    Load a CrossEncoder model for passage reranking.

    Args:
        model_name: HuggingFace model path.
        device:     ``"cuda"`` or ``"cpu"``.

    Returns:
        Loaded ``CrossEncoder`` instance.
    """
    return CrossEncoder(model_name, device=device)


def rerank_documents(
    query: str,
    retrieved_chunks: list[ChunkDict],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[ChunkDict]:
    """
    Score and sort *retrieved_chunks* using a CrossEncoder, return top-*k*.

    Each returned chunk gains a ``"rerank_score"`` key (higher = more relevant).

    Args:
        query:            The user's query string.
        retrieved_chunks: List of chunk dicts (must each have a ``"text"`` key).
        reranker:         Loaded ``CrossEncoder`` from :func:`load_reranker`.
        top_k:            Number of top-scored chunks to return.

    Returns:
        Top-*k* chunks sorted by ``rerank_score`` descending.
    """
    if not retrieved_chunks:
        return []

    pairs  = [[query, c["text"]] for c in retrieved_chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(retrieved_chunks, scores):
        chunk["rerank_score"] = float(score)

    return sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

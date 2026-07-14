"""
chunking.py
===========
PDF ingestion and text chunking strategies for the ML-Bot RAG pipeline.

Strategies (in order of complexity):
    1. Fixed-size   – chunk_pdf_pages()
    2. Recursive    – recursive_chunk_pdf_pages()
    3. Semantic     – semantic_chunk_pdf_pages()
    4. Hybrid       – hybrid_chunk_pdf_pages()   ← production default

Quick start:
    from chunking import open_and_read_pdf, hybrid_chunk_pdf_pages

    pages = open_and_read_pdf("book.pdf")
    chunks = hybrid_chunk_pdf_pages(pages)
"""

from __future__ import annotations

import random
import textwrap
from typing import Any

import fitz  # PyMuPDF
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PageDict  = dict[str, Any]   # output row from open_and_read_pdf
ChunkDict = dict[str, Any]   # output row from any chunking function


# ---------------------------------------------------------------------------
# 1.  PDF ingestion
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse newlines and strip surrounding whitespace."""
    return text.replace("\n", " ").strip()


def open_and_read_pdf(pdf_path: str) -> list[PageDict]:
    """
    Extract per-page text from a PDF with basic statistics.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of page dicts.  Each dict contains:
            - page_number          (int)  adjusted for front-matter offset
            - page_char_count      (int)
            - page_word_count      (int)
            - page_sentence_count_raw (int)
            - page_token_count     (float)  ≈ chars / 4
            - text                 (str)
    """
    doc = fitz.open(pdf_path)
    pages: list[PageDict] = []

    for page_number, page in tqdm(enumerate(doc), total=len(doc), desc="Reading PDF"):
        text = _clean(page.get_text())
        pages.append(
            {
                "page_number":            page_number - 8,   # offset for front matter
                "page_char_count":        len(text),
                "page_word_count":        len(text.split()),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count":       round(len(text) / 4, 2),
                "text":                   text,
            }
        )

    return pages


# ---------------------------------------------------------------------------
# 2.  Fixed-size chunking
# ---------------------------------------------------------------------------

def _fixed_split(text: str, chunk_size: int = 500) -> list[str]:
    """Split *text* into word-boundary chunks of at most *chunk_size* chars."""
    chunks: list[str] = []
    current = ""

    for word in text.split():
        if len(current) + len(word) + 1 < chunk_size:
            current = (current + " " + word) if current else word
        else:
            if current:
                chunks.append(current.strip())
            current = word

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_pdf_pages(
    pages: list[PageDict],
    chunk_size: int = 500,
) -> list[ChunkDict]:
    """
    Fixed-size chunking — split every page on word boundaries.

    Args:
        pages:      Output of :func:`open_and_read_pdf`.
        chunk_size: Maximum characters per chunk.

    Returns:
        List of chunk dicts (page_number, chunk_id, char/word/sentence/token
        counts, text).
    """
    all_chunks: list[ChunkDict] = []

    for page in pages:
        for i, chunk in enumerate(_fixed_split(page["text"], chunk_size)):
            all_chunks.append(
                {
                    "page_number":            page["page_number"],
                    "chunk_id":               i,
                    "page_char_count":        len(chunk),
                    "page_word_count":        len(chunk.split()),
                    "page_sentence_count_raw": len(chunk.split(". ")),
                    "page_token_count":       round(len(chunk) / 4, 2),
                    "text":                   chunk,
                }
            )

    return all_chunks


# ---------------------------------------------------------------------------
# 3.  Recursive chunking
# ---------------------------------------------------------------------------

def _recursive_split(
    text: str,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Hierarchically split *text*: paragraph → line → sentence → word.
    Trailing chunks smaller than *min_chunk_size* are merged upward.
    """

    def _split(chunk: str) -> list[str]:
        # Base case
        if len(chunk) <= max_chunk_size:
            return [chunk.strip()]

        # Level 1: paragraph boundaries
        sections = chunk.split("\n\n")
        if len(sections) > 1:
            out: list[str] = []
            for s in sections:
                if s.strip():
                    out.extend(_split(s.strip()))
            return out

        # Level 2: line boundaries
        sections = chunk.split("\n")
        if len(sections) > 1:
            out = []
            for s in sections:
                if s.strip():
                    out.extend(_split(s.strip()))
            return out

        # Level 3: sentence boundaries
        sentences = nltk.sent_tokenize(chunk)
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent) + 1
            if current_len + sent_len <= max_chunk_size:
                current.append(sent)
                current_len += sent_len
            else:
                if current:
                    pieces.append(" ".join(current))

                if sent_len > max_chunk_size:
                    # Level 4: word-level fallback for giant sentences
                    tmp: list[str] = []
                    tmp_len = 0
                    for word in sent.split():
                        wl = len(word) + 1
                        if tmp_len + wl <= max_chunk_size:
                            tmp.append(word)
                            tmp_len += wl
                        else:
                            pieces.append(" ".join(tmp))
                            tmp, tmp_len = [word], wl
                    if tmp:
                        pieces.append(" ".join(tmp))
                    current, current_len = [], 0
                else:
                    current, current_len = [sent], sent_len

        if current:
            pieces.append(" ".join(current))

        # Merge chunks that are too small
        merged: list[str] = []
        buf = ""
        for piece in pieces:
            if len(buf) + len(piece) < min_chunk_size:
                buf = (buf + " " + piece).lstrip()
            else:
                if buf:
                    merged.append(buf.strip())
                buf = piece
        if buf:
            merged.append(buf.strip())

        return merged

    return _split(text)


def recursive_chunk_pdf_pages(
    pages: list[PageDict],
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
) -> list[ChunkDict]:
    """
    Recursive chunking — hierarchy-aware splitting of every page.

    Args:
        pages:          Output of :func:`open_and_read_pdf`.
        max_chunk_size: Maximum characters per chunk.
        min_chunk_size: Minimum characters; smaller pieces are merged.

    Returns:
        List of chunk dicts.
    """
    all_chunks: list[ChunkDict] = []

    for page in pages:
        for i, chunk in enumerate(
            _recursive_split(page["text"], max_chunk_size, min_chunk_size)
        ):
            all_chunks.append(
                {
                    "page_number":            page["page_number"],
                    "chunk_id":               i,
                    "page_char_count":        len(chunk),
                    "page_word_count":        len(chunk.split()),
                    "page_sentence_count_raw": len(nltk.sent_tokenize(chunk)),
                    "page_token_count":       round(len(chunk) / 4, 2),
                    "text":                   chunk,
                }
            )

    return all_chunks


# ---------------------------------------------------------------------------
# 4.  Semantic chunking
# ---------------------------------------------------------------------------

def _semantic_split(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float = 0.8,
) -> list[str]:
    """
    Group consecutive sentences whose embeddings are similar enough to merge.
    A new chunk begins whenever cosine similarity drops below *similarity_threshold*.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences)
    chunks: list[str] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = float(cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0])
        if sim >= similarity_threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]

    chunks.append(" ".join(current))
    return chunks


def semantic_chunk_pdf_pages(
    pages: list[PageDict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
    similarity_threshold: float = 0.8,
) -> list[ChunkDict]:
    """
    Semantic chunking — split at sentence-boundary semantic breaks.

    Args:
        pages:                Output of :func:`open_and_read_pdf`.
        model_name:           HuggingFace sentence-encoder model path.
        device:               ``"cuda"`` or ``"cpu"``.
        similarity_threshold: Cosine similarity below which a new chunk starts.

    Returns:
        List of chunk dicts.
    """
    model = SentenceTransformer(model_name, device=device)
    all_chunks: list[ChunkDict] = []

    for page in pages:
        for i, chunk in enumerate(
            _semantic_split(page["text"], model, similarity_threshold)
        ):
            all_chunks.append(
                {
                    "page_number":            page["page_number"],
                    "chunk_id":               i,
                    "page_char_count":        len(chunk),
                    "page_word_count":        len(chunk.split()),
                    "page_sentence_count_raw": len(chunk.split(". ")),
                    "page_token_count":       round(len(chunk) / 4, 2),
                    "text":                   chunk,
                }
            )

    return all_chunks


# ---------------------------------------------------------------------------
# 5.  Hybrid chunking  (production default)
# ---------------------------------------------------------------------------

def _hybrid_split(
    text: str,
    model: SentenceTransformer,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
    similarity_threshold: float = 0.7,
) -> list[str]:
    """
    Two-pass split:
        Pass 1 – recursive split at half the target size (creates small, clean pieces).
        Pass 2 – greedily merge adjacent pieces that are semantically similar
                  AND whose combined length stays within *max_chunk_size*.

    Embedding of the current merged chunk is tracked as a running average.
    """
    base = _recursive_split(text, max_chunk_size // 2, min_chunk_size)
    if not base:
        return []

    embeddings = model.encode(base)
    result: list[str] = []
    cur_text = base[0]
    cur_emb  = embeddings[0]

    for i in range(1, len(base)):
        sim = float(cosine_similarity([cur_emb], [embeddings[i]])[0][0])
        fits = len(cur_text) + len(base[i]) <= max_chunk_size

        if sim >= similarity_threshold and fits:
            cur_text = cur_text + " " + base[i]
            cur_emb  = (cur_emb + embeddings[i]) / 2.0
        else:
            result.append(cur_text.strip())
            cur_text = base[i]
            cur_emb  = embeddings[i]

    result.append(cur_text.strip())
    return result


def hybrid_chunk_pdf_pages(
    pages: list[PageDict],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cuda",
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
    similarity_threshold: float = 0.7,
) -> list[ChunkDict]:
    """
    Hybrid chunking — recursive base split merged by semantic similarity.

    This is the **production default** used by the Streamlit app.

    Args:
        pages:                Output of :func:`open_and_read_pdf`.
        model_name:           HuggingFace model used for similarity during merge.
        device:               ``"cuda"`` or ``"cpu"``.
        max_chunk_size:       Maximum characters per final chunk.
        min_chunk_size:       Minimum characters for recursive base chunks.
        similarity_threshold: Cosine similarity threshold for merging.

    Returns:
        List of chunk dicts.
    """
    model = SentenceTransformer(model_name, device=device)
    all_chunks: list[ChunkDict] = []

    for page in pages:
        for i, chunk in enumerate(
            _hybrid_split(
                page["text"], model, max_chunk_size, min_chunk_size, similarity_threshold
            )
        ):
            all_chunks.append(
                {
                    "page_number":            page["page_number"],
                    "chunk_id":               i,
                    "page_char_count":        len(chunk),
                    "page_word_count":        len(chunk.split()),
                    "page_sentence_count_raw": len(nltk.sent_tokenize(chunk)),
                    "page_token_count":       round(len(chunk) / 4, 2),
                    "text":                   chunk,
                }
            )

    return all_chunks


# ---------------------------------------------------------------------------
# Debugging helper
# ---------------------------------------------------------------------------

def visualize_chunks(
    chunks: list[ChunkDict],
    num_samples: int = 5,
    width: int = 80,
) -> None:
    """Pretty-print a random sample of chunks with their metadata."""
    if not chunks:
        print("No chunks to display.")
        return

    samples = random.sample(chunks, min(num_samples, len(chunks)))
    print("\n" + "=" * 80)
    print(f"  CHUNK SAMPLE  ({len(samples)} of {len(chunks)} total)")
    print("=" * 80)

    for idx, chunk in enumerate(samples, start=1):
        print(f"\n── Chunk {idx} {'─' * 60}")
        print(f"  Page    : {chunk.get('page_number', 'N/A')}")
        print(f"  ID      : {chunk.get('chunk_id',   'N/A')}")
        print(f"  Chars   : {chunk.get('page_char_count',  '?')}")
        print(f"  Words   : {chunk.get('page_word_count',  '?')}")
        print(f"  Tokens~ : {chunk.get('page_token_count', '?')}")
        print(f"\n  Text:\n  {textwrap.fill(chunk['text'], width=width, subsequent_indent='  ')}")

    print("\n" + "=" * 80 + "\n")

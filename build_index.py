"""
build_index.py
==============
One-time pipeline script: PDF → chunks → embeddings → FAISS indexes.

Run this **once** before launching the Streamlit app.  All generated
artifacts are saved to the current directory and read back by app.py
via the cached loaders in embedding.py.

Usage:
    python build_index.py --pdf "machine learning book.pdf"

Generated files:
    chunk_sets.pkl                         (chunked pages)
    all_embeddings_results.pkl             (numpy embeddings)
    recursive_mpnet_faiss_index.bin
    recursive_bge_faiss_index.bin
    recursive_minilm_faiss_index.bin
    hybrid_mpnet_faiss_index.bin
    hybrid_bge_faiss_index.bin
    hybrid_minilm_faiss_index.bin
"""

from __future__ import annotations

import argparse
import time

import torch

from chunking import (
    open_and_read_pdf,
    hybrid_chunk_pdf_pages,
    recursive_chunk_pdf_pages,
)
from embedding import (
    build_and_save_indexes,
    generate_embeddings_for_all,
    load_embedding_models,
    save_chunk_sets,
    save_embeddings,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chunk sets, embeddings, and FAISS indexes for ML-Bot."
    )
    parser.add_argument(
        "--pdf",
        default="machine learning book.pdf",
        help="Path to the source PDF file  (default: 'machine learning book.pdf')",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Compute device for embedding models  (default: cuda if available)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk  (default: 1000)",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Minimum characters before merging  (default: 100)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 60)
    print("  ML-Bot  ·  Index Builder")
    print("=" * 60)
    print(f"  PDF    : {args.pdf}")
    print(f"  Device : {args.device}")
    print()

    # ── 1. Read PDF ────────────────────────────────────────────────────────
    t0 = time.time()
    print("── Step 1 / 4  Reading PDF …")
    pages = open_and_read_pdf(args.pdf)
    print(f"   {len(pages)} pages extracted  ({time.time() - t0:.1f}s)\n")

    # ── 2. Chunk ───────────────────────────────────────────────────────────
    print("── Step 2 / 4  Chunking …")

    t1 = time.time()
    recursive_chunks = recursive_chunk_pdf_pages(
        pages,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
    )
    print(f"   Recursive : {len(recursive_chunks)} chunks  ({time.time() - t1:.1f}s)")

    t1 = time.time()
    hybrid_chunks = hybrid_chunk_pdf_pages(
        pages,
        device         = args.device,
        max_chunk_size = args.max_chunk_size,
        min_chunk_size = args.min_chunk_size,
    )
    print(f"   Hybrid    : {len(hybrid_chunks)} chunks  ({time.time() - t1:.1f}s)\n")

    chunk_sets = {
        "recursive": recursive_chunks,
        "hybrid":    hybrid_chunks,
    }
    save_chunk_sets(chunk_sets)

    # ── 3. Embed ───────────────────────────────────────────────────────────
    print("── Step 3 / 4  Generating embeddings …")
    t2 = time.time()

    loaded_models = load_embedding_models(device=args.device)
    all_embeddings_results = generate_embeddings_for_all(chunk_sets, loaded_models)
    save_embeddings(all_embeddings_results)
    print(f"   Done  ({time.time() - t2:.1f}s)\n")

    # ── 4. Build FAISS indexes ─────────────────────────────────────────────
    print("── Step 4 / 4  Building FAISS indexes …")
    t3 = time.time()
    build_and_save_indexes(all_embeddings_results)
    print(f"   Done  ({time.time() - t3:.1f}s)\n")

    # ── Summary ────────────────────────────────────────────────────────────
    total = time.time() - t0
    print("=" * 60)
    print(f"  ✅  All artifacts generated in {total:.1f}s")
    print("     chunk_sets.pkl")
    print("     all_embeddings_results.pkl")
    print("     recursive_{{mpnet,bge,minilm}}_faiss_index.bin")
    print("     hybrid_{{mpnet,bge,minilm}}_faiss_index.bin")
    print()
    print("  Run the app with:  streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

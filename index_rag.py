"""
RAG Indexing Script
===================
Indexes the `rag-only` clinical dataset (or legacy PDF) for retrieval.

Usage:
  # Local SQLite FTS5 verification (instant, zero-cost, works immediately):
  python index_rag.py --backend fts

  # Compile Markdown Knowledge Base:
  python index_rag.py --backend compile

  # Index into local ChromaDB with embeddings:
  python index_rag.py --backend local

  # Upload to OpenAI Vector Store (cloud production):
  python index_rag.py --backend openai
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description="Index fetal ultrasound RAG data from rag-only dataset.")
parser.add_argument(
    "--data-dir",
    default="rag-only",
    help="Path to the rag-only dataset directory (default: 'rag-only')"
)
parser.add_argument(
    "--backend",
    choices=["fts", "local", "openai", "compile"],
    default="fts",
    help="Indexing backend: 'fts' (SQLite FTS5), 'local' (ChromaDB), 'openai' (Cloud Vector Store), 'compile' (Markdown KB)"
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Force reindexing even if already built"
)
parser.add_argument(
    "--pdf",
    required=False,
    default=None,
    help="Legacy PDF path fallback"
)

args = parser.parse_args()
data_path = Path(args.data_dir)

if not data_path.exists() and not args.pdf:
    print(f"❌ Error: Data directory '{args.data_dir}' not found.")
    sys.exit(1)

if args.backend == "compile":
    from pipeline2_rag import compile_rag_knowledge_base
    out = compile_rag_knowledge_base(data_path)
    print(f"\n✅ Markdown knowledge base successfully compiled to:\n   {out}")

elif args.backend == "fts":
    from pipeline2_rag import retrieve_sqlite_fts
    print(f"[RAG] Verifying SQLite FTS5 index at '{data_path / 'indexes' / 'kb.sqlite'}' …")
    test_results = retrieve_sqlite_fts("four chamber view normal cardiac anatomy", k=2, data_dir=data_path)
    if test_results:
        print(f"✅ SQLite FTS5 index is operational! Found {len(test_results)} test matches.")
        print("Sample record preview:\n" + test_results[0][:250] + "…")
    else:
        print("⚠️ Warning: FTS5 query returned no results. Verify that kb.sqlite exists in rag-only/indexes/.")

elif args.backend == "local":
    from pipeline2_rag import build_local_vector_store
    print(f"[RAG] Building persistent ChromaDB index from '{data_path}' …")
    build_local_vector_store(data_path, force_reindex=args.force)
    print("\n✅ Local ChromaDB index ready at ./chroma_db")

elif args.backend == "openai":
    from pipeline2_rag import build_openai_vector_store
    print(f"[RAG] Building OpenAI Cloud Vector Store from '{data_path}' …")
    vs_id = build_openai_vector_store(data_path)
    print(f"\n✅ OpenAI Vector Store ready!")
    print(f"Add this to your .env file:\nOPENAI_VECTOR_STORE_ID={vs_id}")

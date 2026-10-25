"""
One-time RAG indexing script.
Run this ONCE before starting the server.

Usage:
  python index_rag.py --pdf "path/to/ISUOG_2023_Cardiac_Screening.pdf" --backend openai
  python index_rag.py --pdf "path/to/ISUOG_2023_Cardiac_Screening.pdf" --backend local

The --backend openai option uploads the PDF to the OpenAI vector store
and prints the vector_store_id to add to your .env file.
"""

import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--pdf",     required=True, help="Path to the ISUOG PDF")
parser.add_argument("--backend", choices=["openai", "local"], default="openai")
args = parser.parse_args()

if args.backend == "openai":
    from pipeline2_rag import build_openai_vector_store
    vs_id = build_openai_vector_store(args.pdf)
    print(f"\n✅ Add this to your .env:\nOPENAI_VECTOR_STORE_ID={vs_id}")
else:
    from pipeline2_rag import build_local_vector_store
    build_local_vector_store(args.pdf)
    print("\n✅ Local ChromaDB index built at ./chroma_db")

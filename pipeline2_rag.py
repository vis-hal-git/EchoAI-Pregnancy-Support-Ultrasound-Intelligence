"""
Pipeline 2 – Inbuilt Doctor Knowledge Base (RAG)
==================================================
Offline indexing:
  ISUOG PDF → PyPDFLoader → RecursiveCharacterTextSplitter (800 / 150)
  → OpenAI text-embedding-3-small → OpenAI Vector Store (Files API)

Runtime retrieval:
  User query → query expansion → search OpenAI Vector Store (k=4)
  → returns top-K chunk strings fed into Pipeline 3 prompt

Uses the OpenAI native vector-store (Files API + vector_stores) so
no local ChromaDB process is required in production. A local ChromaDB
fallback is also provided for offline development.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
EMBED_MODEL      = "text-embedding-3-small"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K            = 4
VS_NAME          = "ultrasound_isuog_rag"          # display name in OpenAI dashboard
VS_ID_ENV        = "OPENAI_VECTOR_STORE_ID"        # set this after first upload


client = OpenAI(api_key=OPENAI_API_KEY)


# ── Chunking helper (no LangChain dependency for the RAG upload path) ──────
def _split_text(text: str,
                chunk_size: int   = CHUNK_SIZE,
                overlap: int      = CHUNK_OVERLAP) -> list[str]:
    """
    Naive recursive character splitter:
    tries to break on double-newline, then newline, then space, then char.
    """
    separators = ["\n\n", "\n", " ", ""]
    def _split(t: str, seps: list[str]) -> list[str]:
        if not seps or len(t) <= chunk_size:
            return [t] if t.strip() else []
        sep = seps[0]
        parts = t.split(sep)
        chunks: list[str] = []
        buf = ""
        for part in parts:
            candidate = (buf + sep + part).lstrip(sep) if buf else part
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    buf = ""
                else:
                    buf = part
        if buf.strip():
            chunks.append(buf.strip())
        # add overlap
        result: list[str] = []
        for i, c in enumerate(chunks):
            if i > 0 and overlap > 0:
                prev_tail = chunks[i - 1][-overlap:]
                result.append(prev_tail + "\n" + c)
            else:
                result.append(c)
        return result
    return _split(text, separators)


# ── PDF text extraction ────────────────────────────────────────────────────
def _extract_pdf_text(pdf_path: str) -> str:
    """Extract raw text from a PDF using pypdf (pure-Python, no poppler)."""
    try:
        from pypdf import PdfReader          # pip install pypdf
    except ImportError:
        from PyPDF2 import PdfReader         # older name fallback
    reader = PdfReader(pdf_path)
    pages  = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


# ══════════════════════════════════════════════════════════════════════════
# OPTION A – OpenAI Vector Store  (production)
# ══════════════════════════════════════════════════════════════════════════
def build_openai_vector_store(pdf_path: str) -> str:
    """
    Upload the PDF to OpenAI Files API, create a Vector Store, attach the
    file, and wait for indexing to complete.

    Returns the vector_store_id – save it as OPENAI_VECTOR_STORE_ID in .env
    so you never re-upload.
    """
    pdf_path = str(Path(pdf_path).expanduser().resolve())
    print(f"[RAG] Uploading {pdf_path} to OpenAI Files API …")

    with open(pdf_path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="assistants")
    file_id = uploaded.id
    print(f"[RAG] File uploaded  → file_id={file_id}")

    vs_api = getattr(client, "vector_stores", getattr(client.beta, "vector_stores", None))
    # Create (or reuse) a named vector store
    vs = vs_api.create(name=VS_NAME)
    vs_id = vs.id
    print(f"[RAG] Vector store created → vs_id={vs_id}")

    # Attach the file and wait for indexing
    vs_api.files.create(vector_store_id=vs_id, file_id=file_id)
    print("[RAG] Waiting for indexing …", end="", flush=True)
    while True:
        status = vs_api.files.list(vector_store_id=vs_id).data
        states = [f.status for f in status]
        if all(s == "completed" for s in states):
            break
        if any(s == "failed" for s in states):
            raise RuntimeError("Vector store indexing failed.")
        print(".", end="", flush=True)
        time.sleep(2)
    print(f"\n[RAG] Indexing complete.  Add to .env:  {VS_ID_ENV}={vs_id}")
    return vs_id


def retrieve_openai(query: str,
                    vs_id: Optional[str] = None,
                    k: int = TOP_K) -> list[str]:
    """
    Query the OpenAI vector store.
    Returns a list of chunk strings (top-k by cosine similarity).
    """
    vs_id = vs_id or os.environ.get(VS_ID_ENV)
    if not vs_id:
        raise ValueError(
            f"No vector store ID.  Set {VS_ID_ENV} in .env or call "
            "build_openai_vector_store() first."
        )

    vs_api = getattr(client, "vector_stores", getattr(client.beta, "vector_stores", None))
    if vs_api and hasattr(vs_api, "search"):
        try:
            res = vs_api.search(vector_store_id=vs_id, query=_expand_query(query), max_num_results=k)
            chunks: list[str] = []
            for item in res.data:
                content = getattr(item, "content", None)
                if content:
                    if isinstance(content, list):
                        for c in content:
                            text = getattr(c, "text", str(c))
                            if text and text not in chunks:
                                chunks.append(text)
                    elif isinstance(content, str):
                        chunks.append(content)
            if chunks:
                return chunks[:k]
        except Exception as e:
            print(f"[RAG] vector_stores.search failed: {e}")

    # Fallback to Assistants path
    chunks = _retrieve_via_assistant(query, vs_id, k)
    return chunks[:k]


def _expand_query(query: str) -> str:
    """Prepend clinical context to improve embedding recall."""
    return (
        "Obstetric, gynecological and fetal ultrasound clinical guidelines: " + query
    )


def _retrieve_via_assistant(query: str, vs_id: str, k: int) -> list[str]:
    """
    Fallback: create a temporary Assistant with file_search, run a thread,
    parse the annotations.
    """
    asst = client.beta.assistants.create(
        name         = "_rag_retriever_tmp",
        model        = "gpt-4o-mini",
        instructions = "Return the most relevant guideline excerpts verbatim.",
        tools        = [{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
    )
    thread = client.beta.threads.create(
        messages=[{"role": "user",
                   "content": _expand_query(query)}]
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id    = thread.id,
        assistant_id = asst.id,
    )
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    chunks: list[str] = []
    for msg in msgs.data:
        if msg.role == "assistant":
            for block in msg.content:
                if block.type == "text":
                    # grab citation annotations
                    for ann in getattr(block.text, "annotations", []):
                        if ann.type == "file_citation":
                            chunks.append(getattr(ann, "text", ""))
                    if not chunks:
                        chunks.append(block.text.value)
    # Clean up temporary assistant
    client.beta.assistants.delete(asst.id)
    return chunks[:k]


# ══════════════════════════════════════════════════════════════════════════
# OPTION B – Local ChromaDB  (offline / development)
# ══════════════════════════════════════════════════════════════════════════
CHROMA_DIR = "./chroma_db"

def build_local_vector_store(pdf_path: str) -> None:
    """
    Chunk the PDF locally and store in a persistent ChromaDB collection.
    Run once; subsequent calls are no-ops if the collection already exists.
    """
    import chromadb                             # pip install chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    chroma  = chromadb.PersistentClient(path=CHROMA_DIR)
    ef      = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
    coll    = chroma.get_or_create_collection("isuog_rag", embedding_function=ef)

    if coll.count() > 0:
        print(f"[RAG-local] Collection already has {coll.count()} chunks. Skipping.")
        return

    raw    = _extract_pdf_text(pdf_path)
    chunks = _split_text(raw)
    ids    = [f"chunk_{i}" for i in range(len(chunks))]
    print(f"[RAG-local] Indexing {len(chunks)} chunks into ChromaDB …")
    # Chroma handles batching internally
    coll.add(documents=chunks, ids=ids)
    print("[RAG-local] Done.")


def retrieve_local(query: str, k: int = TOP_K) -> list[str]:
    """Query the local ChromaDB store."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
    coll   = chroma.get_collection("isuog_rag", embedding_function=ef)
    res    = coll.query(query_texts=[_expand_query(query)], n_results=k)
    return res["documents"][0] if res["documents"] else []


# ── Unified public interface ───────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K) -> list[str]:
    """
    Auto-select retrieval backend:
      • OPENAI_VECTOR_STORE_ID set → OpenAI vector store (production)
      • else                       → local ChromaDB (development)
    """
    vs_id = os.environ.get(VS_ID_ENV, "").strip()
    if vs_id:
        try:
            chunks = retrieve_openai(query, vs_id=vs_id, k=k)
            if chunks:
                return chunks
        except Exception as e:
            print(f"[RAG] OpenAI vector store retrieval failed: {e}. Falling back to local ChromaDB.")
    return retrieve_local(query, k=k)


def format_rag_context(chunks: list[str]) -> str:
    """Format retrieved chunks into a prompt-ready block."""
    if not chunks:
        return ""
    parts = [f"[Guideline excerpt {i+1}]\n{c.strip()}" for i, c in enumerate(chunks)]
    return "RELEVANT ISUOG GUIDELINE EXCERPTS:\n" + "\n\n".join(parts)


# ── CLI helper ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "build-openai":
        pdf = sys.argv[2]
        vid = build_openai_vector_store(pdf)
        print(f"Done. vector_store_id = {vid}")

    elif cmd == "build-local":
        pdf = sys.argv[2]
        build_local_vector_store(pdf)

    elif cmd == "query":
        q = " ".join(sys.argv[2:])
        chunks = retrieve(q)
        for i, c in enumerate(chunks, 1):
            print(f"\n── Chunk {i} ──\n{c[:300]} …")
    else:
        print("Usage:")
        print("  python pipeline2_rag.py build-openai <path/to/isuog.pdf>")
        print("  python pipeline2_rag.py build-local  <path/to/isuog.pdf>")
        print("  python pipeline2_rag.py query <your question here>")

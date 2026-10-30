"""
Pipeline 2 – Inbuilt Doctor Knowledge Base (RAG)
==================================================
Uses the structured clinical RAG dataset in `rag-only`:
  - 1,520 curated fetal ultrasound records across views, findings, diseases,
    workflows, cases, and media.
  - Linked to 1,445 evidence excerpts and 68 primary clinical sources
    (including ISUOG Practice Guidelines: Carvalho 2023, Yoo, Tegnander, Paladini).

Retrieval Architecture:
  1. Direct SQLite FTS5 (Instant, zero-dependency, local BM25 ranking across records,
     evidence, and guideline citations in `rag-only/indexes/kb.sqlite`).
  2. Local ChromaDB (Semantic vector search over structured clinical chunks with
     OpenAI text-embedding-3-small).
  3. OpenAI Vector Store (Production cloud index via OpenAI Files API).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Paths & Configuration ──────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent.resolve()
RAG_DIR           = Path(os.getenv("RAG_DIR", str(BASE_DIR / "rag-only")))
RECORDS_DIR       = RAG_DIR / "records"
INDEXES_DIR       = RAG_DIR / "indexes"
DB_PATH           = INDEXES_DIR / "kb.sqlite"
COMPILED_KB_PATH  = RAG_DIR / "fetal_cardiac_rag_knowledge_base.md"

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL       = "text-embedding-3-small"
CHROMA_DIR        = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION = "fetal_rag_kb"
TOP_K             = 4
VS_NAME           = "fetal_cardiac_rag_kb"
VS_ID_ENV         = "OPENAI_VECTOR_STORE_ID"

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# In-memory caches for fast clinical citation resolution
_sources_cache: Optional[dict[str, dict]] = None
_evidence_cache: Optional[dict[str, dict]] = None


# ── Metadata Loading Helpers ───────────────────────────────────────────────
def load_sources_and_evidence(rag_dir: Path | str = RAG_DIR) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load sources.jsonl and evidence.jsonl into memory for fast citation lookups."""
    global _sources_cache, _evidence_cache
    if _sources_cache is not None and _evidence_cache is not None:
        return _sources_cache, _evidence_cache

    rdir = Path(rag_dir)
    src_path = rdir / "records" / "sources.jsonl"
    ev_path = rdir / "records" / "evidence.jsonl"

    sources: dict[str, dict] = {}
    if src_path.exists():
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    sources[item.get("paper_id", "")] = item

    evidence: dict[str, dict] = {}
    if ev_path.exists():
        with open(ev_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    evidence[item.get("evidence_id", "")] = item

    _sources_cache = sources
    _evidence_cache = evidence
    return _sources_cache, _evidence_cache


# ── Knowledge Base Compilation Helper ──────────────────────────────────────
def compile_rag_knowledge_base(data_dir: Path | str = RAG_DIR) -> Path:
    """
    Compiles all structured records from `rag-only` into a comprehensive,
    human- and LLM-readable Markdown knowledge base.
    Used for OpenAI Vector Store uploads and local inspections.
    """
    rdir = Path(data_dir)
    out_path = rdir / "fetal_cardiac_rag_knowledge_base.md"
    sources, evidence = load_sources_and_evidence(rdir)

    lines = [
        "# CLINICAL KNOWLEDGE BASE: FETAL ULTRASOUND SCREENING & CARDIAC ANATOMY",
        "Source: ISUOG Practice Guidelines (Carvalho 2023) and Multi-Institutional Ultrasound Archive.",
        "",
        "---",
        ""
    ]

    record_files = [
        ("views.jsonl", "1. STANDARD ULTRASOUND VIEWS & ANATOMICAL PLANES"),
        ("workflows.jsonl", "2. CLINICAL WORKFLOWS & SCREENING PROTOCOLS"),
        ("diseases.jsonl", "3. CONGENITAL HEART DISEASES & PATHOLOGIES"),
        ("findings.jsonl", "4. SONOGRAPHIC FINDINGS & DIAGNOSTIC FEATURES"),
        ("cases.jsonl", "5. CLINICAL CASES & SONOGRAPHIC PRESENTATIONS"),
        ("media.jsonl", "6. ULTRASOUND IMAGING SCANS & FIGURE EVIDENCE"),
    ]

    for rf, section_title in record_files:
        path = rdir / "records" / rf
        if not path.exists():
            continue

        lines.append(f"## {section_title}")
        lines.append("")

        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                rec = json.loads(l)
                label = rec.get("label", "Record")
                kind = rf.split(".")[0].rstrip("s")
                lines.append(f"### [{kind.upper()}] {label}")

                meta = []
                if "modality" in rec: meta.append(f"**Modality**: {rec['modality']}")
                if "gestational_age" in rec: meta.append(f"**Gestational Age**: {rec['gestational_age']}")
                if "caption" in rec: meta.append(f"**Scan Caption**: {rec['caption']}")
                if "demonstrated_findings" in rec:
                    meta.append(f"**Demonstrated Findings**: {', '.join(rec['demonstrated_findings'])}")
                if meta:
                    lines.append(" - " + " | ".join(meta))

                for c in rec.get("claims", []):
                    pred = c.get("predicate", "").replace("_", " ").title()
                    val = c.get("value", "")
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    lines.append(f"- **{pred}**: {val}")

                # Linked evidence excerpt
                ev_excerpts = []
                for eid in rec.get("evidence_ids", []):
                    if eid in evidence:
                        ev = evidence[eid]
                        text = ev.get("original_excerpt") or ev.get("paraphrase_en")
                        if text:
                            ev_excerpts.append(text)
                if ev_excerpts:
                    lines.append(f"- **Clinical Evidence**: *\"{ev_excerpts[0]}\"*")

                # Linked citation
                cits = []
                for pid in rec.get("source_paper_ids", []):
                    if pid in sources:
                        s = sources[pid]
                        author = s.get("authors", ["Unknown"])[0]
                        cits.append(f"{author} et al. ({s.get('year', '')}), *{s.get('title', '')}* [{s.get('journal', '')}]")
                if cits:
                    lines.append(f"- **Guideline / Reference Citation**: {'; '.join(cits[:2])}")

                lines.append("")

    # Full Bibliography
    lines.append("## 7. PRIMARY CLINICAL SOURCES & GUIDELINE BIBLIOGRAPHY")
    lines.append("")
    for pid, s in sources.items():
        authors = ", ".join(s.get("authors", []))
        lines.append(f"- **[{s.get('year', '')}] {s.get('title', '')}**")
        lines.append(f"  Authors: {authors}")
        lines.append(f"  Journal: {s.get('journal', '')} | DOI: {s.get('doi', 'N/A')}")
        lines.append("")

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[RAG] Compiled knowledge base generated at: {out_path} ({len(content):,} chars)")
    return out_path


# ── Chunk Generator for Vector Indexes ─────────────────────────────────────
def load_clinical_chunks(data_dir: Path | str = RAG_DIR) -> list[tuple[str, str, dict]]:
    """
    Produce rich structured clinical chunks for ChromaDB and Vector Stores.
    Returns list of (chunk_id, chunk_text, metadata_dict).
    """
    rdir = Path(data_dir)
    sources, evidence = load_sources_and_evidence(rdir)
    chunks: list[tuple[str, str, dict]] = []

    record_files = ["views.jsonl", "diseases.jsonl", "findings.jsonl", "workflows.jsonl", "cases.jsonl", "media.jsonl"]

    for rf in record_files:
        path = rdir / "records" / rf
        if not path.exists():
            continue
        kind = rf.split(".")[0]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("id", f"{kind}_{len(chunks)}")
                label = rec.get("label", "Record")

                claims_text = []
                for c in rec.get("claims", []):
                    pred = c.get("predicate", "").replace("_", " ").title()
                    val = c.get("value", "")
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    claims_text.append(f"- {pred}: {val}")

                extra = []
                if "modality" in rec: extra.append(f"Modality: {rec['modality']}")
                if "gestational_age" in rec: extra.append(f"Gestational age: {rec['gestational_age']}")
                if "caption" in rec: extra.append(f"Caption: {rec['caption']}")
                if "demonstrated_findings" in rec:
                    extra.append(f"Demonstrated findings: {', '.join(rec['demonstrated_findings'])}")

                cits = []
                for pid in rec.get("source_paper_ids", []):
                    if pid in sources:
                        s = sources[pid]
                        first_author = s.get("authors", ["Unknown"])[0]
                        cits.append(f"{first_author} et al. ({s.get('year', '')}), \"{s.get('title', '')}\" [{s.get('journal', '')}]")

                evs = []
                for eid in rec.get("evidence_ids", []):
                    if eid in evidence:
                        ev = evidence[eid]
                        orig = ev.get("original_excerpt") or ev.get("paraphrase_en") or ""
                        if orig:
                            evs.append(f'"{orig}"')

                doc_parts = [f"[{kind.upper()}] {label}"]
                if extra:
                    doc_parts.append(" | ".join(extra))
                if claims_text:
                    doc_parts.append("\n".join(claims_text))
                if evs:
                    doc_parts.append("Clinical Evidence: " + " | ".join(evs[:2]))
                if cits:
                    doc_parts.append("Guideline Citations:\n  • " + "\n  • ".join(cits[:2]))

                chunk_text = "\n".join(doc_parts)
                meta = {
                    "kind": kind,
                    "label": label,
                    "primary_citation": cits[0] if cits else "ISUOG Guidelines Archive"
                }
                chunks.append((rid, chunk_text, meta))

    return chunks


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL ENGINE 1 – Direct SQLite FTS5 (Zero-dependency & instant)
# ══════════════════════════════════════════════════════════════════════════
def retrieve_sqlite_fts(query: str, k: int = TOP_K, data_dir: Path | str = RAG_DIR) -> list[str]:
    """
    Search `rag-only/indexes/kb.sqlite` using SQLite FTS5 full-text search with BM25.
    Works 100% offline with zero dependencies or API keys.
    """
    db_file = Path(data_dir) / "indexes" / "kb.sqlite"
    if not db_file.exists():
        return []

    sources, evidence = load_sources_and_evidence(data_dir)

    tokens = re.findall(r"[a-zA-Z0-9]+", query)
    if not tokens:
        return []

    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "to", "for", "is", "are",
        "with", "at", "by", "from", "what", "how", "does", "scan", "ultrasound",
        "patient", "can", "you", "tell", "me", "show"
    }
    meaningful = [t for t in tokens if t.lower() not in stopwords] or tokens

    # FTS5 query with prefix wildcards for medical stem matching
    fts_expr = " OR ".join(f'"{t}"*' for t in meaningful[:8])

    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.execute("""
            SELECT r.id, r.kind, r.label, f.content, bm25(records_fts) as rank
            FROM records_fts f
            JOIN records r ON r.id = f.id
            WHERE records_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_expr, k))
        rows = cursor.fetchall()
    except Exception:
        # Fallback to simple phrase search
        try:
            safe_query = " OR ".join(f'"{t}"' for t in meaningful[:5])
            cursor = conn.execute("""
                SELECT r.id, r.kind, r.label, f.content, 0 as rank
                FROM records_fts f
                JOIN records r ON r.id = f.id
                WHERE records_fts MATCH ?
                LIMIT ?
            """, (safe_query, k))
            rows = cursor.fetchall()
        except Exception:
            return []
    finally:
        conn.close()

    chunks = []
    for rid, kind, label, content, rank in rows:
        # Fetch citations from sources
        cits = []
        try:
            conn_sub = sqlite3.connect(str(db_file))
            p_row = conn_sub.execute("SELECT source_paper_ids FROM records WHERE id = ?", (rid,)).fetchone()
            conn_sub.close()
            if p_row and p_row[0]:
                pids = json.loads(p_row[0])
                for pid in pids:
                    if pid in sources:
                        s = sources[pid]
                        first_author = s.get("authors", ["Unknown"])[0]
                        cits.append(f"{first_author} et al. ({s.get('year', '')}), \"{s.get('title', '')}\" [{s.get('journal', '')}]")
        except Exception:
            pass

        doc_parts = [f"[{kind.upper()}] {label}"]
        clean_content = content.strip()
        if clean_content.lower().startswith(label.lower()):
            clean_content = clean_content[len(label):].strip()
        if clean_content:
            doc_parts.append(clean_content)
        if cits:
            doc_parts.append("Guideline Citations:\n  • " + "\n  • ".join(cits[:2]))

        chunks.append("\n".join(doc_parts))

    return chunks


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL ENGINE 2 – Local ChromaDB Vector Store (Semantic Embeddings)
# ══════════════════════════════════════════════════════════════════════════
def build_local_vector_store(data_dir: Path | str = RAG_DIR,
                             collection_name: str = CHROMA_COLLECTION,
                             force_reindex: bool = False) -> None:
    """
    Indexes the 1,520 clinical records from `rag-only` into ChromaDB.
    Batches uploads to OpenAI embeddings in increments of 100 chunks.
    """
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    if OPENAI_API_KEY:
        os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        api_key_env_var="OPENAI_API_KEY",
        model_name=EMBED_MODEL
    )

    if force_reindex:
        try:
            chroma.delete_collection(collection_name)
        except Exception:
            pass

    coll = chroma.get_or_create_collection(collection_name, embedding_function=ef)

    if coll.count() >= 1000 and not force_reindex:
        print(f"[RAG-local] Collection '{collection_name}' already indexed with {coll.count()} chunks. Skipping.")
        return

    print(f"[RAG-local] Loading clinical chunks from {data_dir} …")
    chunks = load_clinical_chunks(data_dir)
    total = len(chunks)
    print(f"[RAG-local] Indexing {total} structured clinical records into ChromaDB ({collection_name}) …")

    batch_size = 100
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = [b[0] for b in batch]
        docs = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"  -> Indexed {min(i + batch_size, total)} / {total} chunks …")

    print(f"[RAG-local] ChromaDB indexing complete. Total records: {coll.count()}.")


def retrieve_chroma(query: str, k: int = TOP_K, collection_name: str = CHROMA_COLLECTION) -> list[str]:
    """Query ChromaDB semantic index."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    if OPENAI_API_KEY:
        os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        api_key_env_var="OPENAI_API_KEY",
        model_name=EMBED_MODEL
    )
    try:
        coll = chroma.get_collection(collection_name, embedding_function=ef)
        if coll.count() == 0:
            return []
        res = coll.query(query_texts=[_expand_query(query)], n_results=k)
        return res["documents"][0] if res.get("documents") else []
    except Exception as e:
        return []


def retrieve_local(query: str, k: int = TOP_K) -> list[str]:
    """
    Hybrid local retrieval:
    Merges ChromaDB semantic embeddings with SQLite FTS5 exact clinical term matching.
    Falls back gracefully to SQLite FTS5 if ChromaDB is not yet populated.
    """
    results: list[str] = []

    # 1. Try ChromaDB
    chroma_chunks = retrieve_chroma(query, k=k)
    for c in chroma_chunks:
        if c not in results:
            results.append(c)

    # 2. Add / augment with SQLite FTS5
    fts_chunks = retrieve_sqlite_fts(query, k=k)
    for c in fts_chunks:
        if c not in results:
            results.append(c)

    return results[:k] if results else fts_chunks[:k]


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL ENGINE 3 – OpenAI Vector Store (Production Cloud)
# ══════════════════════════════════════════════════════════════════════════
def build_openai_vector_store(data_dir: Path | str = RAG_DIR) -> str:
    """
    Compile `rag-only` into a structured clinical reference markdown guide,
    upload to OpenAI Files API, and create an OpenAI Vector Store.
    """
    if not client:
        raise ValueError("OPENAI_API_KEY not set.")

    kb_file = compile_rag_knowledge_base(data_dir)
    print(f"[RAG] Uploading compiled knowledge base {kb_file} to OpenAI Files API …")

    with open(kb_file, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="assistants")
    file_id = uploaded.id
    print(f"[RAG] File uploaded -> file_id={file_id}")

    vs_api = getattr(client, "vector_stores", getattr(client.beta, "vector_stores", None))
    vs = vs_api.create(name=VS_NAME)
    vs_id = vs.id
    print(f"[RAG] Vector store created -> vs_id={vs_id}")

    vs_api.files.create(vector_store_id=vs_id, file_id=file_id)
    print("[RAG] Waiting for OpenAI vector indexing …", end="", flush=True)
    while True:
        status = vs_api.files.list(vector_store_id=vs_id).data
        states = [f.status for f in status]
        if all(s == "completed" for s in states):
            break
        if any(s == "failed" for s in states):
            raise RuntimeError("OpenAI Vector store indexing failed.")
        print(".", end="", flush=True)
        time.sleep(2)

    print(f"\n[RAG] Indexing complete! Add to .env: {VS_ID_ENV}={vs_id}")
    return vs_id


def retrieve_openai(query: str,
                    vs_id: Optional[str] = None,
                    k: int = TOP_K) -> list[str]:
    """Query the OpenAI Vector Store."""
    if not client:
        return []

    vs_id = vs_id or os.environ.get(VS_ID_ENV, "").strip()
    if not vs_id:
        return []

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
            print(f"[RAG] vector_stores.search error: {e}")

    # Fallback to Assistants file_search
    return _retrieve_via_assistant(query, vs_id, k)


def _retrieve_via_assistant(query: str, vs_id: str, k: int) -> list[str]:
    """Fallback file_search via OpenAI Assistants."""
    if not client:
        return []
    asst = client.beta.assistants.create(
        name="_rag_retriever_tmp",
        model="gpt-4o-mini",
        instructions="Return the most relevant guideline excerpts verbatim.",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
    )
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": _expand_query(query)}]
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=asst.id,
    )
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    chunks: list[str] = []
    for msg in msgs.data:
        if msg.role == "assistant":
            for block in msg.content:
                if block.type == "text":
                    for ann in getattr(block.text, "annotations", []):
                        if ann.type == "file_citation":
                            chunks.append(getattr(ann, "text", ""))
                    if not chunks:
                        chunks.append(block.text.value)
    client.beta.assistants.delete(asst.id)
    return chunks[:k]


def _expand_query(query: str) -> str:
    """Prepend clinical context to improve semantic recall."""
    return f"Fetal ultrasound obstetric cardiac guidelines: {query}"


# ── Unified Public Interface ───────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K) -> list[str]:
    """
    Unified public retrieval entry point:
      • If OPENAI_VECTOR_STORE_ID is set in .env -> queries OpenAI Vector Store
      • Else -> queries local hybrid RAG (ChromaDB + SQLite FTS5) on `rag-only`
    """
    vs_id = os.environ.get(VS_ID_ENV, "").strip()
    if vs_id and client:
        try:
            chunks = retrieve_openai(query, vs_id=vs_id, k=k)
            if chunks:
                return chunks
        except Exception as e:
            print(f"[RAG] OpenAI vector store retrieval error: {e}. Falling back to local RAG.")

    return retrieve_local(query, k=k)


def format_rag_context(chunks: list[str]) -> str:
    """Format retrieved clinical records into a system prompt knowledge block."""
    if not chunks:
        return ""
    parts = [f"[Guideline / Reference {i+1}]\n{c.strip()}" for i, c in enumerate(chunks)]
    return "RELEVANT CLINICAL GUIDELINE & ULTRASOUND EVIDENCE:\n" + "\n\n".join(parts)


def extract_cited_excerpts(rag_chunks: list[str]) -> list[dict]:
    """Return structured citation metadata for UI rendering and API responses."""
    citations = []
    for i, c in enumerate(rag_chunks):
        lines = [line.strip() for line in c.strip().split("\n") if line.strip()]
        header = lines[0] if lines else f"Guideline {i+1}"
        citations.append({
            "id": i + 1,
            "label": header,
            "text": c.strip()[:600]
        })
    return citations


# ── CLI Helper ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "build-openai":
        target = sys.argv[2] if len(sys.argv) > 2 else str(RAG_DIR)
        vid = build_openai_vector_store(target)
        print(f"Done. Vector store ID = {vid}")

    elif cmd == "build-local":
        target = sys.argv[2] if len(sys.argv) > 2 else str(RAG_DIR)
        build_local_vector_store(target)

    elif cmd == "compile-kb":
        target = sys.argv[2] if len(sys.argv) > 2 else str(RAG_DIR)
        out = compile_rag_knowledge_base(target)
        print(f"Compiled KB created at: {out}")

    elif cmd == "query":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "four chamber view normal anatomy"
        chunks = retrieve(q)
        print(f"\nRetrieved {len(chunks)} chunks for query: '{q}'\n")
        for i, c in enumerate(chunks, 1):
            print(f"── Record {i} ──\n{c[:400]}\n")
    else:
        print("Usage:")
        print("  python pipeline2_rag.py build-local    [data_dir (default: rag-only)]")
        print("  python pipeline2_rag.py build-openai   [data_dir (default: rag-only)]")
        print("  python pipeline2_rag.py compile-kb     [data_dir (default: rag-only)]")
        print("  python pipeline2_rag.py query <question>")

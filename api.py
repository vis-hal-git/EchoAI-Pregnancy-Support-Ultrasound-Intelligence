"""
FastAPI backend
================
POST /analyze   – one-shot analysis (image or video + query)
POST /chat      – follow-up turn on an existing session
GET  /session/{sid}/citations – return RAG citations for a session
GET  /health    – liveness probe
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from pipeline1_ingestion import IMAGE_EXTS, VIDEO_EXTS, ingest_image, ingest_video
from pipeline2_rag import retrieve
from pipeline3_fusion import extract_cited_excerpts, run_inference

load_dotenv()

app = FastAPI(title="Ultrasound AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (swap for Redis in production) ─────────────────
# { session_id: { "history": [...], "rag_chunks": [...] } }
_sessions: dict[str, dict] = {}

DEFAULT_K_FRAMES = int(os.getenv("DEFAULT_K_FRAMES", "8"))


def _ingest_upload(upload: UploadFile, k: int) -> list:
    ext = os.path.splitext(upload.filename)[1].lower()
    if ext in IMAGE_EXTS:
        return ingest_image(upload.file.read())
    if ext in VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(upload.file.read())
            tmp_path = tmp.name
        try:
            return ingest_video(tmp_path, k=k)
        finally:
            os.remove(tmp_path)
    raise HTTPException(400, f"Unsupported file type: {ext}")


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None, description="Optional scan (.png) or sweep (.mp4)"),
    query: str                 = Form(..., description="Pregnancy support query or consultation question"),
    k: int                     = Form(DEFAULT_K_FRAMES, description="Keyframes (video only)"),
):
    """
    Full pipeline: optional scan ingestion → RAG retrieve → GPT-4o → response.
    Supports both multimodal ultrasound scans and text-only pregnancy support queries.
    Returns a new session_id for follow-up /chat calls.
    """
    frames = []
    is_video = False
    if file and file.filename:
        frames = _ingest_upload(file, k)
        is_video = bool(frames and frames[0].timestamp_s is not None)

    rag_chunks = retrieve(query)
    answer     = run_inference(frames, query, rag_chunks)

    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "history": [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ],
        "rag_chunks": rag_chunks,
        "frames": frames,
    }

    return {
        "session_id": sid,
        "answer": answer,
        "citations": extract_cited_excerpts(rag_chunks),
        "frame_count": len(frames),
        "is_video": is_video,
    }


@app.post("/chat")
async def chat(
    session_id: str            = Form(...),
    follow_up: str             = Form(..., description="Follow-up question"),
    file: Optional[UploadFile] = File(None, description="Optional new scan"),
    k: int                     = Form(DEFAULT_K_FRAMES),
):
    """
    Continue a conversation started by /analyze.
    Re-uses the cached RAG chunks and ultrasound frames from that session.
    Optionally accepts a new scan for comparison.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found. Call /analyze first.")

    # Retrieve relevant RAG chunks for follow-up and merge with session chunks
    new_chunks = retrieve(follow_up)
    existing_chunks = session.get("rag_chunks", [])
    for c in new_chunks:
        if c not in existing_chunks:
            existing_chunks.append(c)
    rag_chunks = existing_chunks
    session["rag_chunks"] = rag_chunks
    history = session["history"]

    if file and file.filename:
        frames = _ingest_upload(file, k)
        session["frames"] = frames
    else:
        frames = session.get("frames", [])

    answer = run_inference(frames, follow_up, rag_chunks, history)

    session["history"].append({"role": "user",      "content": follow_up})
    session["history"].append({"role": "assistant",  "content": answer})

    return {
        "session_id": session_id,
        "answer": answer,
        "citations": extract_cited_excerpts(rag_chunks),
    }


@app.get("/session/{session_id}/citations")
def get_citations(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    return {"citations": extract_cited_excerpts(session["rag_chunks"])}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}

"""
Pipeline 1 – Visual & Temporal Ingestion
=========================================
Path A (static image / u2-bench):  RGB → Base64 JPEG
Path B (ultrasound video .mp4):    cv2.VideoCapture → uniform temporal
                                   sampling → K timestamped frames → Base64

Returns list[Frame] consumed by Pipeline 3 via to_openai_content().
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from dataclasses import dataclass
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

# ── Constants ──────────────────────────────────────────────────────────────
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_K   = 8       # keyframes sampled from a video
MAX_SIDE    = 1024    # longest pixel side sent to GPT-4o  (token budget)
JPEG_Q      = 90


# ── Data model ─────────────────────────────────────────────────────────────
@dataclass
class Frame:
    b64: str                           # Base64-encoded JPEG (no data-URI prefix)
    index: int   | None = None         # source frame index  (video only)
    timestamp_s: float | None = None   # wall-clock seconds  (video only)
    mime: str = "image/jpeg"

    @property
    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.b64}"


# ── Shared helpers ─────────────────────────────────────────────────────────
def _resize(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1.0:
        return img
    return img.resize((int(w * s), int(h * s)), Image.LANCZOS)


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    _resize(img).save(buf, format="JPEG", quality=JPEG_Q)
    return base64.b64encode(buf.getvalue()).decode()


# ── Path A: static image ───────────────────────────────────────────────────
def ingest_image(source: str | bytes | BinaryIO) -> list[Frame]:
    """Single scan (.png etc.) → RGB → Base64.  Returns [Frame]."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    return [Frame(b64=_pil_to_b64(img))]


# ── Path B: video ──────────────────────────────────────────────────────────
def _uniform_indices(n: int, k: int) -> list[int]:
    k = max(1, min(k, n))
    if k == 1:
        return [n // 2]
    return [int(i) for i in np.linspace(0, n - 1, k).round()]


def _bgr_to_frame(bgr: np.ndarray, idx: int, fps: float) -> Frame:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Frame(b64=_pil_to_b64(Image.fromarray(rgb)),
                 index=idx, timestamp_s=round(idx / fps, 2))


def ingest_video(path: str, k: int = DEFAULT_K) -> list[Frame]:
    """Ultrasound sweep → cv2.VideoCapture → uniform sample → K timestamped frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:               # some containers don't report frame count
            return _sequential_sample(cap, fps, k)
        frames: list[Frame] = []
        for idx in _uniform_indices(total, k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok:
                frames.append(_bgr_to_frame(bgr, idx, fps))
        if not frames:
            raise ValueError("No readable frames in video.")
        return frames
    finally:
        cap.release()


def _sequential_sample(cap: cv2.VideoCapture, fps: float, k: int) -> list[Frame]:
    """Fallback: read all frames then down-sample."""
    all_bgr: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        all_bgr.append(bgr)
    if not all_bgr:
        raise ValueError("Empty video.")
    return [_bgr_to_frame(all_bgr[i], i, fps)
            for i in _uniform_indices(len(all_bgr), k)]


# ── Router ─────────────────────────────────────────────────────────────────
def ingest(path: str, k: int = DEFAULT_K) -> list[Frame]:
    """Route a file path to Path A or Path B."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return ingest_image(path)
    if ext in VIDEO_EXTS:
        return ingest_video(path, k)
    raise ValueError(f"Unsupported extension: {ext}")


def ingest_upload(uploaded_file, k: int = DEFAULT_K) -> list[Frame]:
    """Accept a Streamlit UploadedFile (image or video)."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in IMAGE_EXTS:
        return ingest_image(uploaded_file.getvalue())
    if ext in VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            return ingest_video(tmp_path, k)
        finally:
            os.remove(tmp_path)
    raise ValueError(f"Unsupported extension: {ext}")


# ── Hand-off to Pipeline 3 ─────────────────────────────────────────────────
def to_openai_content(frames: list[Frame], detail: str = "high") -> list[dict]:
    """
    Build the image portion of a GPT-4o user message.
    Video → ordered (timestamp-text, image) pairs so the model
    sees the temporal sweep explicitly.
    """
    if not frames:
        return []

    content: list[dict] = []
    is_video = frames[0].timestamp_s is not None

    if is_video:
        content.append({"type": "text",
                        "text": (f"The following {len(frames)} images are keyframes "
                                 "sampled uniformly from one ultrasound sweep, in order.")})
    for n, f in enumerate(frames, 1):
        if is_video:
            content.append({"type": "text",
                            "text": f"Frame {n}/{len(frames)}  t = {f.timestamp_s:.2f}s"})
        content.append({"type": "image_url",
                        "image_url": {"url": f.data_uri, "detail": detail}})
    return content

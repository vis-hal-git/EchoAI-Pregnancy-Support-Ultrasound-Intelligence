"""
Streamlit UI – WombCare Ultrasound Multimodal AI
================================================
Redesigned to match WombCare branding:
Warm rose/blush palette, soft white cards, feminine medical aesthetic.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import time
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.jpeg")
LOGO_B64 = ""
if os.path.exists(LOGO_PATH):
    try:
        with open(LOGO_PATH, "rb") as f:
            LOGO_B64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WombCare | Pregnancy Support & Fetal Ultrasound Intelligence",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── WombCare-Inspired CSS ─────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Global Reset & Base Typography ── */
:root {
    --text-color: #1e1b2e !important;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMarkdownContainer"],
.stMarkdown {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #1e1b2e !important;
}

/* Force dark ink contrast on all text elements across themes */
p, span, div, li, ol, ul, strong, em, b, i,
h1, h2, h3, h4, h5, h6,
label, .stMarkdown {
    color: #1e1b2e !important;
}

/* ── WombCare Token System ── */
:root {
    --rose:        #e91e8c;
    --rose-dark:   #c0176f;
    --rose-soft:   #f472b6;
    --blush:       #fce7f3;
    --blush-light: #fdf2f8;
    --lavender:    #ede9fe;
    --lavender-mid:#c4b5fd;
    --sage:        #d1fae5;
    --sage-mid:    #6ee7b7;
    --amber:       #fef3c7;
    --amber-mid:   #fcd34d;
    --ink:         #1e1b2e;
    --ink-mid:     #4b4569;
    --ink-light:   #7c7a8e;
    --border:      #f0e6f6;
    --bg:          #fdf8ff;
    --card:        #ffffff;
    --shadow:      0 2px 16px rgba(233,30,140,0.07), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-hover: 0 6px 28px rgba(233,30,140,0.13), 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Page background ── */
.stApp {
    background: var(--bg);
}

/* ── Streamlit block cleanup ── */
section[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}

/* ── Top Header ── */
.wc-header {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.wc-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.wc-logo {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, var(--rose) 0%, #9333ea 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}

.wc-logo-img {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    object-fit: cover;
    box-shadow: var(--shadow);
    border: 1.5px solid var(--border);
    flex-shrink: 0;
    background: #ffffff;
}

.wc-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0;
}

.wc-subtitle {
    font-size: 0.8rem;
    color: var(--ink-light);
    margin-top: 2px;
    font-weight: 400;
}

/* ── Pill badges ── */
.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 11px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}

.badge-online {
    background: var(--sage);
    color: #065f46;
    border: 1px solid #6ee7b7;
}

.badge-offline {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
}

.badge-guide {
    background: var(--blush);
    color: var(--rose-dark);
    border: 1px solid #fbcfe8;
}

.badge-model {
    background: var(--lavender);
    color: #5b21b6;
    border: 1px solid var(--lavender-mid);
}

.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

.dot-on  { background: #10b981; box-shadow: 0 0 6px #10b981; animation: pulse 2s infinite; }
.dot-off { background: #ef4444; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Section heading ── */
.sec-heading {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── Panel card ── */
.wc-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, border-color 0.2s;
}

.wc-card:hover {
    border-color: #fbcfe8;
    box-shadow: var(--shadow-hover);
}

/* ── Scan container ── */
.scan-wrap {
    border: 2px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: #f9f5ff;
    box-shadow: var(--shadow);
}

/* ── Classification banners ── */
.banner {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    font-weight: 500;
}

.banner-gravid {
    background: var(--sage);
    border-left: 5px solid #059669;
    color: #064e3b;
}

.banner-nongravid {
    background: var(--blush);
    border-left: 5px solid var(--rose);
    color: var(--rose-dark);
}

.banner-indeterminate {
    background: var(--amber);
    border-left: 5px solid #d97706;
    color: #78350f;
}

.banner-info {
    background: var(--lavender);
    border-left: 5px solid #7c3aed;
    color: #3b0764;
}

.banner-title {
    font-weight: 700;
    margin-bottom: 3px;
    font-size: 0.88rem;
}

/* ── Citation cards ── */
.citation {
    background: var(--blush-light);
    border: 1px solid #fbcfe8;
    border-left: 4px solid var(--rose);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
    color: var(--ink-mid);
    font-size: 0.86rem;
    line-height: 1.55;
}

.citation-label {
    font-weight: 700;
    color: var(--rose-dark);
    font-size: 0.78rem;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Quick preset label ── */
.preset-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--ink-light);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── Disclaimer ── */
.disclaimer {
    background: var(--blush-light);
    border: 1px dashed #fbcfe8;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: var(--ink-light);
    font-size: 0.76rem;
    line-height: 1.5;
    margin-top: 1.25rem;
}

/* ── Streamlit button adjustments ── */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.18s;
    border: 1.5px solid transparent;
}

div.stButton > button[kind="secondary"],
div.stButton > button[kind="secondary"] * {
    background: var(--blush) !important;
    color: var(--rose-dark) !important;
    border-color: #fbcfe8 !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: #fce7f3 !important;
    border-color: var(--rose-soft) !important;
    transform: translateY(-1px);
    box-shadow: 0 3px 12px rgba(233,30,140,0.15);
}

div.stButton > button[kind="primary"],
div.stButton > button[kind="primary"] * {
    background: linear-gradient(135deg, var(--rose) 0%, #9333ea 100%) !important;
    color: #ffffff !important;
    border: none !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(233,30,140,0.35);
}

div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] > button * {
    background: var(--blush) !important;
    color: var(--rose-dark) !important;
    border: 1.5px solid #fbcfe8 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    background: #fce7f3 !important;
    border-color: var(--rose-soft) !important;
}

/* ── Tab styling ── */
div[data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--blush-light);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}

div[data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.84rem;
    color: var(--ink-light) !important;
}

div[data-baseweb="tab"] * {
    color: var(--ink-light) !important;
}

div[aria-selected="true"][data-baseweb="tab"] {
    background: var(--card) !important;
    box-shadow: var(--shadow);
}

div[aria-selected="true"][data-baseweb="tab"] * {
    color: var(--rose) !important;
}

/* ── Chat bubbles ── */
div[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 2px 10px rgba(233, 30, 140, 0.05) !important;
    color: #1e1b2e !important;
}

div[data-testid="stChatMessage"] * {
    color: #1e1b2e !important;
}

div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
    color: #1e1b2e !important;
}

div[data-testid="stChatMessage"] code {
    background: #fce7f3 !important;
    color: #be185d !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-size: 0.86em !important;
    font-weight: 600 !important;
}

/* Distinct subtle tinting for user vs assistant messages */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #fdf2f8 !important;
    border: 1px solid #fbcfe8 !important;
}

div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #ffffff !important;
    border: 1px solid #ede9fe !important;
    border-left: 4px solid var(--rose) !important;
}

/* ── Tab Content & Report Markdown ── */
div[data-testid="stTabContent"] {
    color: #1e1b2e !important;
}

div[data-testid="stTabContent"] [data-testid="stMarkdownContainer"] * {
    color: #1e1b2e !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: var(--blush-light) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
    color: var(--ink-light) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: var(--rose-dark) !important;
}

/* ── Text Inputs & Textarea ── */
.stTextArea textarea, .stTextInput input {
    color: #1e1b2e !important;
    background-color: #ffffff !important;
    border-color: #f0e6f6 !important;
}

.stTextArea label, .stTextInput label {
    color: #1e1b2e !important;
    font-weight: 600 !important;
}

/* ── Alert boxes ── */
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
    color: inherit !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #4b4569 !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--rose-dark) !important;
    font-size: 0.95rem;
    font-weight: 700;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}

section[data-testid="stSidebar"] caption,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #7c7a8e !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
INITIAL_STATE = {
    "session_id": None,
    "chat_history": [],
    "citations": [],
    "frame_count": 0,
    "is_video": False,
    "uploaded_filename": None,
    "query_input": "",
    "backend_online": False,
    "last_latency_ms": None,
}
for key, default_val in INITIAL_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# ── Backend Health ─────────────────────────────────────────────────────────────
def check_backend_health() -> tuple[bool, Optional[int]]:
    t0 = time.time()
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2.0)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return True, latency
    except Exception:
        pass
    return False, None


is_online, latency = check_backend_health()
st.session_state.backend_online = is_online
st.session_state.last_latency_ms = latency


# ── Header ────────────────────────────────────────────────────────────────────
if is_online:
    status_badge = f'<span class="badge badge-online"><span class="dot dot-on"></span>Online · {latency}ms</span>'
else:
    status_badge = '<span class="badge badge-offline"><span class="dot dot-off"></span>Offline</span>'

logo_display = (
    f'<img src="data:image/jpeg;base64,{LOGO_B64}" class="wc-logo-img" alt="WombCare Logo" />'
    if LOGO_B64 else '<div class="wc-logo">🌸</div>'
)

st.markdown(f"""
<div class="wc-header">
    <div class="wc-brand">
        {logo_display}
        <div>
            <div class="wc-title">WombCare · Pregnancy & Ultrasound Intelligence</div>
            <div class="wc-subtitle">Multimodal fetal assessment grounded in clinical ultrasound data (`rag-only`)</div>
        </div>
    </div>
    <div class="badge-row">
        {status_badge}
        <span class="badge badge-guide">RAG Knowledge Base</span>
        <span class="badge badge-model">GPT-4o Vision</span>
        <span class="badge badge-model">Multimodal AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not is_online:
    st.error(f"⚠️ Backend unreachable at `{API_URL}`. Ensure `uvicorn api:app --port 8000` is running.")


# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1.95], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Image Upload & Analysis Action
# ══════════════════════════════════════════════════════════════════════════════
with left_col:

    st.markdown('<div class="sec-heading">📁 Ultrasound Image Upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.84rem; color:#7c7a8e; margin-top:-4px; margin-bottom:12px;">'
        'Upload your ultrasound image below, then click <strong>Upload & Generate Report</strong> to receive a detailed clinical assessment.'
        '</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload ultrasound image (.png, .jpg, .jpeg) or cine sweep (.mp4, .avi)",
        type=["png","jpg","jpeg","bmp","tif","tiff","webp","mp4","avi","mov","mkv"],
        help="Select a 2D/3D ultrasound image or sweep file for clinical assessment.",
        label_visibility="collapsed",
    )

    is_video_upload = False
    if uploaded:
        st.session_state.uploaded_filename = uploaded.name
        is_video_upload = uploaded.type.startswith("video") or uploaded.name.lower().endswith((".mp4",".avi",".mov",".mkv"))
        st.markdown('<div class="scan-wrap">', unsafe_allow_html=True)
        if is_video_upload:
            st.video(uploaded)
        else:
            st.image(uploaded, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        file_size_kb = len(uploaded.getvalue()) / 1024
        size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{(file_size_kb/1024):.2f} MB"
        st.caption(f"🖼️ **{uploaded.name}** · `{size_str}` · `{'Video cine sweep' if is_video_upload else 'Ultrasound scan'}`")
    else:
        st.markdown("""
        <div style="border: 2px dashed #f0e6f6; border-radius: 12px; padding: 2rem 1.2rem; text-align: center; background: #ffffff; margin-bottom: 1rem; box-shadow: var(--shadow);">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🩺</div>
            <div style="font-weight: 700; color: #1e1b2e; font-size: 0.95rem;">Select an Ultrasound Scan or Image</div>
            <div style="font-size: 0.8rem; color: #7c7a8e; margin-top: 4px;">Supports standard formats (.png, .jpg, .jpeg)</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("💬 Optional: Custom Focus or Clinical Query", expanded=False):
        custom_question = st.text_input(
            "Specific area to evaluate (optional)",
            value="",
            placeholder="e.g. Check for gestational sac viability, evaluate endometrial stripe, or review ovaries",
            help="Leave blank to automatically run the full standard clinical assessment report.",
        )
        if is_video_upload:
            k_frames = st.slider("Cine sweep keyframes (k)", 4, 16, 8, 1, help="Number of frames extracted across the cine-loop sweep.")
        else:
            k_frames = 8

    b1, b2 = st.columns([2.3, 1])
    with b1:
        generate_clicked = st.button("🌸 Upload & Generate Report", type="primary", use_container_width=True)
    with b2:
        reset_clicked = st.button("↩ Reset", use_container_width=True)

    if reset_clicked:
        if st.session_state.session_id:
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
        for k in ["session_id","chat_history","citations","frame_count","is_video","uploaded_filename","query_input"]:
            st.session_state[k] = ([] if k in ("chat_history","citations") else
                                   False if k == "is_video" else
                                   0 if k == "frame_count" else
                                   "" if k == "query_input" else None)
        st.success("Session cleared.")
        st.rerun()

    if generate_clicked:
        if not uploaded:
            st.warning("⚠️ Please select and upload an ultrasound image first.")
        elif not is_online:
            st.error("⚠️ Backend is offline — cannot run analysis.")
        else:
            default_eval_prompt = (
                "Perform a comprehensive ultrasound evaluation: "
                "1. Determine and classify whether this scan indicates PREGNANCY DETECTED (GRAVID) "
                "or NON-PREGNANT (NON-GRAVID) or INDETERMINATE (PUL / Early Correlation). "
                "2. Provide a detailed anatomical breakdown and checklist of all visible pelvic and fetal structures. "
                "3. Assess the ultrasound plane, acoustic quality, and imaging orientation. "
                "4. Correlate with clinical reference standards from the knowledge base and provide clear recommendations."
            )
            final_query = custom_question.strip() if (custom_question and custom_question.strip()) else default_eval_prompt

            with st.status("Analyzing ultrasound image & generating report…", expanded=True) as status_box:
                t_start = time.time()
                status_box.write("Consulting clinical knowledge base (`rag-only`)…")
                time.sleep(0.1)
                status_box.write("Ingesting scan & processing keyframes…")
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = requests.post(
                        f"{API_URL}/analyze",
                        files=files,
                        data={"query": final_query, "k": k_frames if "k_frames" in locals() else 8},
                        timeout=140,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id   = data["session_id"]
                        st.session_state.is_video     = data.get("is_video", False)
                        st.session_state.frame_count  = data.get("frame_count", 0)
                        st.session_state.citations    = data.get("citations", [])
                        now_str = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history = [
                            {"role": "user",      "content": f"Ultrasound Image Analysis: {uploaded.name}", "timestamp": now_str},
                            {"role": "assistant", "content": data["answer"],     "timestamp": now_str},
                        ]
                        elapsed = time.time() - t_start
                        status_box.update(label=f"✅ Detailed report ready in {elapsed:.1f}s", state="complete", expanded=False)
                        st.rerun()
                    else:
                        status_box.update(label="Analysis error", state="error")
                        st.error(f"API {resp.status_code}: {resp.text}")
                except Exception as e:
                    status_box.update(label="Connection error", state="error")
                    st.error(f"Could not reach backend: {e}")

    st.markdown("""
    <div class="disclaimer">
        <strong>⚖️ Clinical support tool only.</strong> WombCare output is grounded in the structured clinical ultrasound
        knowledge base (`rag-only`) and is intended for educational and decision-support purposes. It does not replace
        assessment by a certified sonographer or fetal medicine specialist.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Report, Chat, Citations, Telemetry
# ══════════════════════════════════════════════════════════════════════════════
with right_col:

    if st.session_state.session_id and st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else ""
        if "PREGNANCY DETECTED" in latest or "PREGNANT" in latest:
            st.markdown('<div class="banner banner-gravid"><div class="banner-title">🤰 Pregnancy Detected</div>Obstetric structures identified and grounded in clinical ultrasound records.</div>', unsafe_allow_html=True)
        elif "NON-PREGNANT" in latest or "NON-GRAVID" in latest:
            st.markdown('<div class="banner banner-nongravid"><div class="banner-title">🩺 Non-Gravid Scan</div>Pelvic anatomy identified without intrauterine gestational sac.</div>', unsafe_allow_html=True)
        elif "INDETERMINATE" in latest:
            st.markdown('<div class="banner banner-indeterminate"><div class="banner-title">⚠️ Indeterminate — Correlation Required</div>Equivocal findings. Clinical correlation with serum β-hCG and repeat TVS advised.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner banner-info"><div class="banner-title">💡 Pregnancy Support Consultation</div>Evidence-based response grounded in clinical ultrasound knowledge base.</div>', unsafe_allow_html=True)

    tab_report, tab_chat, tab_cites, tab_tele = st.tabs([
        "📋 Detailed Assessment Report",
        "💬 Consultation & Follow-up",
        f"📚 Clinical Records ({len(st.session_state.citations)})",
        "🔬 Session Info",
    ])

    # ── Report Tab ──
    with tab_report:
        if st.session_state.chat_history:
            reports = [t["content"] for t in st.session_state.chat_history if t["role"] == "assistant"]
            if reports:
                latest_report = reports[-1]
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "📥 Download Report (.md)",
                        data=latest_report,
                        file_name=f"wombcare_{st.session_state.session_id[:8] if st.session_state.session_id else 'report'}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "📄 Export Consultation (.json)",
                        data=json.dumps(st.session_state.chat_history, indent=2),
                        file_name=f"wombcare_case_{st.session_state.session_id[:8] if st.session_state.session_id else 'case'}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                st.markdown("### Clinical Assessment")
                st.markdown(latest_report)
        else:
            st.info("👆 Upload an ultrasound image on the left and click **Upload & Generate Report** to produce a detailed assessment.")

    # ── Chat Tab ──
    with tab_chat:
        if not st.session_state.chat_history:
            st.info("Upload an ultrasound image on the left and click **Upload & Generate Report** to begin.")
        else:
            for turn in st.session_state.chat_history:
                role    = turn["role"]
                content = turn["content"]
                ts      = turn.get("timestamp", "")
                if role == "user":
                    with st.chat_message("user", avatar="🌸"):
                        st.markdown(f"**Your scan** `{ts}`")
                        st.markdown(content)
                else:
                    with st.chat_message("assistant", avatar=LOGO_PATH if os.path.exists(LOGO_PATH) else "🩺"):
                        st.markdown(f"**WombCare** `{ts}`")
                        st.markdown(content)

            if st.session_state.session_id:
                st.markdown("---")
                follow_up = st.chat_input("Ask a follow-up question…")
                if follow_up:
                    with st.spinner("Consulting clinical knowledge base…"):
                        try:
                            resp = requests.post(
                                f"{API_URL}/chat",
                                data={
                                    "session_id": st.session_state.session_id,
                                    "follow_up": follow_up,
                                    "k": k_frames if "k_frames" in locals() else 8,
                                },
                                timeout=110,
                            )
                            if resp.status_code == 200:
                                res = resp.json()
                                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                                st.session_state.chat_history.append({"role":"user",      "content":follow_up,      "timestamp":now_str})
                                st.session_state.chat_history.append({"role":"assistant", "content":res["answer"],  "timestamp":now_str})
                                if res.get("citations"):
                                    st.session_state.citations = res["citations"]
                                st.rerun()
                            else:
                                st.error(f"Error {resp.status_code}: {resp.text}")
                        except Exception as e:
                            st.error(f"Chat error: {e}")

    # ── Citations Tab ──
    with tab_cites:
        if st.session_state.citations:
            st.markdown(f"**{len(st.session_state.citations)} records** retrieved from the `rag-only` clinical knowledge base.")
            for c in st.session_state.citations:
                label = c.get("label", f"Record [{c.get('id','•')}]")
                text  = c.get("text", "")
                st.markdown(f"""
                <div class="citation">
                    <div class="citation-label">📖 {label}</div>
                    <div style="white-space:pre-wrap;">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Clinical records will appear here after your first analysis.")

    # ── Telemetry Tab ──
    with tab_tele:
        st.markdown("### Session Diagnostics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Session ID", (st.session_state.session_id[:8]+"…") if st.session_state.session_id else "None")
        m2.metric("Scan type",  "Cine sweep" if st.session_state.is_video else "Static frame")
        m3.metric("Keyframes",  st.session_state.frame_count)
        st.markdown("#### Configuration")
        st.json({
            "API_URL":        API_URL,
            "Backend":        "Online" if st.session_state.backend_online else "Offline",
            "Latency (ms)":   st.session_state.last_latency_ms,
            "Knowledge base": "rag-only dataset · 1,520 records + SQLite FTS5",
            "Vision model":   "GPT-4o Multimodal (High Detail)",
            "Clinical data":  "Curated clinical ultrasound knowledge base (rag-only)",
        })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=175)
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🌸 Clinical Reference")
    st.markdown("""
**WombCare** grounds all responses in the structured clinical ultrasound knowledge base (`rag-only`):

**First Trimester (5–12 wks)**
- 4.5–5 wks: Gestational sac in decidua
- 5–5.5 wks: Yolk sac confirms IUP
- 6–6.5 wks: Fetal pole & cardiac flicker (>100 bpm)
- 7–12 wks: CRL dating ±3–5 days accuracy
- 11–13+6 wks: Nuchal translucency screening

**Second Trimester (18–22 wks)**
- 5-plane fetal anatomical screening:
  · Situs & stomach
  · 4-Chamber view (45° axis)
  · LVOT · RVOT · 3VT
- Biometry: BPD, HC, AC, FL

**Third Trimester (28–40 wks)**
- Fetal growth & estimated weight
- Amniotic fluid: SDP 2–8 cm / AFI 5–25 cm
- Placental localisation & umbilical Doppler
    """)

    st.divider()

    st.markdown("### ⚙️ Pipeline")
    st.markdown("""
- Optical flow keyframe extraction
- ChromaDB / SQLite FTS5 RAG (rag-only)
- GPT-4o multimodal vision fusion
    """)

    st.divider()

    st.markdown("### 🩺 Session")
    if st.session_state.session_id:
        st.caption(f"UUID: `{st.session_state.session_id}`")
        if st.button("Clear session", use_container_width=True):
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
            for k in ["session_id","chat_history","citations","frame_count","is_video","uploaded_filename","query_input"]:
                st.session_state[k] = ([] if k in ("chat_history","citations") else
                                       False if k == "is_video" else
                                       0 if k == "frame_count" else
                                       "" if k == "query_input" else None)
            st.rerun()
    else:
        st.caption("No active session.")

    st.caption("WombCare · v2.1 · Clinical Ultrasound AI")
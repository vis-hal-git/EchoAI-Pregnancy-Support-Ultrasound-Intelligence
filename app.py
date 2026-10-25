"""
Streamlit UI – Ultrasound Multimodal AI
========================================
Executive Medical Dashboard for Fetal Cardiac & Pelvic Ultrasound Analysis
Powered by ISUOG 2023 Guidelines & GPT-4o Multimodal Vision
"""

from __future__ import annotations

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

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EchoAI | Pregnancy Support & Fetal Ultrasound Intelligence",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Medical Executive CSS Styling ──────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Typography & Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* ── Custom Theme Palette & Background ── */
:root {
    --bg-dark: #090e17;
    --card-bg: rgba(17, 24, 39, 0.75);
    --card-border: rgba(56, 189, 248, 0.15);
    --card-border-hover: rgba(56, 189, 248, 0.45);
    --cyan-accent: #00f0ff;
    --sky-accent: #38bdf8;
    --emerald-accent: #10b981;
    --amber-accent: #f59e0b;
    --rose-accent: #f43f5e;
    --indigo-accent: #6366f1;
}

/* ── Top Header Navigation Bar ── */
.nav-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px -5px rgba(56, 189, 248, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(16px);
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 16px;
}

.brand-icon {
    font-size: 2.4rem;
    background: linear-gradient(135deg, #ff4b4b 0%, #f43f5e 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 8px rgba(244, 63, 94, 0.4));
    animation: heartBeat 2s infinite ease-in-out;
}

@keyframes heartBeat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.18); }
    30% { transform: scale(1); }
    45% { transform: scale(1.1); }
}

.brand-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.15;
}

.brand-subtitle {
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 3px;
    letter-spacing: 0.02em;
}

/* ── Badges ── */
.badge-container {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.status-online {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.status-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

.dot-online {
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: blink 2s infinite ease-in-out;
}

.dot-offline {
    background-color: #ef4444;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.guideline-tag {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
}

.model-tag {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Panel Card Containers ── */
.panel-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease-in-out;
}

.panel-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
}

.panel-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Clinical Overview Alert Banners ── */
.alert-gravid {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
    border-left: 5px solid #10b981;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    color: #ecfdf5;
}

.alert-nongravid {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(30, 58, 138, 0.25) 100%);
    border-left: 5px solid #38bdf8;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    color: #f0f9ff;
}

.alert-indeterminate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(120, 53, 15, 0.25) 100%);
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    color: #fffbeb;
}

/* ── Citations & Medical Cards ── */
.citation-chip {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.5;
}

.citation-header {
    font-weight: 700;
    color: #38bdf8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

/* ── Quick Question Pills ── */
.quick-prompt-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* ── Disclaimer Box ── */
.disclaimer-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px dashed rgba(148, 163, 184, 0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #94a3b8;
    font-size: 0.77rem;
    line-height: 1.45;
    margin-top: 1.5rem;
}

/* ── St Button Tweaks ── */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px -2px rgba(56, 189, 248, 0.3);
}

/* ── Scan Inspection Frame ── */
.scan-container {
    border: 2px solid rgba(56, 189, 248, 0.3);
    border-radius: 12px;
    overflow: hidden;
    background: #000;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.6);
}
</style>
""", unsafe_allow_html=True)


# ── Session State Management ──────────────────────────────────────────────────
INITIAL_STATE = {
    "session_id": None,
    "chat_history": [],      # [{"role": "user"|"assistant", "content": str, "timestamp": str}]
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


# ── Backend Health Check Function ─────────────────────────────────────────────
def check_backend_health() -> tuple[bool, Optional[int]]:
    """Probe the FastAPI server /health endpoint and return (is_online, latency_ms)."""
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


# ── Top Navigation Header ─────────────────────────────────────────────────────
status_html = f"""
<div class="status-pill status-online">
    <span class="status-dot dot-online"></span>
    Backend Online ({latency}ms)
</div>
""" if is_online else """
<div class="status-pill status-offline">
    <span class="status-dot dot-offline"></span>
    Backend Offline
</div>
"""

st.markdown(f"""
<div class="nav-header">
    <div class="brand-wrapper">
        <div class="brand-icon">🤰</div>
        <div>
            <div class="brand-title">EchoAI · Pregnancy Support & Ultrasound Intelligence</div>
            <div class="brand-subtitle">Multimodal Fetal Assessment, Maternal Support Queries & ISUOG Guideline Grounding</div>
        </div>
    </div>
    <div class="badge-container">
        {status_html}
        <div class="guideline-tag">ISUOG Practice Standards</div>
        <div class="model-tag">Text & Multimodal Query</div>
        <div class="model-tag">GPT-4o Vision & Knowledge</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not is_online:
    st.error(
        f"⚠️ **FastAPI Backend Unreachable at `{API_URL}`**. "
        "Please ensure `python -m uvicorn api:app --port 8000` is running in the background."
    )


# ── Main Two-Column Layout ────────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1.95], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN: Media Ingestion & Query Configuration
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    # ── Upload Section ──
    st.markdown('<div class="panel-heading">📁 Ultrasound Scan / Sweep Ingestion <span style="font-size:0.75rem; font-weight:normal; color:#94a3b8;">(Optional)</span></div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload Ultrasound Cine Sweep (.mp4, .avi) or Static Frame (.png, .jpg) — Optional",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "mp4", "avi", "mov", "mkv"],
        help="Upload scan/sweep for visual evaluation, or leave blank to ask text pregnancy support queries.",
        label_visibility="collapsed",
    )

    is_video_upload = False
    if uploaded:
        st.session_state.uploaded_filename = uploaded.name
        is_video_upload = uploaded.type.startswith("video") or uploaded.name.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )

        st.markdown('<div class="scan-container">', unsafe_allow_html=True)
        if is_video_upload:
            st.video(uploaded)
        else:
            st.image(uploaded, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # File Telemetry Badge
        file_size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.caption(f"📄 **{uploaded.name}** | Size: `{file_size_mb:.2f} MB` | Format: `{'Video Sweep' if is_video_upload else 'Static Scan'}`")

    # ── Sweep Parameters Accordion ──
    with st.expander("⚙️ Sweep & Vision Processing Parameters", expanded=False):
        k_frames = st.slider(
            "Video Keyframes Extraction (k)",
            min_value=4,
            max_value=16,
            value=8,
            step=1,
            help="ISUOG cardiac sweep analysis extracts k keyframes via optical flow / uniformity sampling.",
        )
        st.caption("ℹ️ *Default 8 frames covers complete cardiac cycle (4CH → LVOT → RVOT → 3VT)*.")

    # ── Clinical Preset Chips ──
    st.markdown('<div class="quick-prompt-label">⚡ Fast Pregnancy Consultation Queries</div>', unsafe_allow_html=True)
    preset_cols = st.columns(2)
    
    with preset_cols[0]:
        if st.button("🤰 Early Viability & Sac", use_container_width=True, help="Key markers for early pregnancy confirmation"):
            st.session_state.query_input = (
                "What are the critical ultrasound markers to confirm intrauterine pregnancy viability "
                "(gestational sac, double decidual sign, yolk sac, fetal pole, embryonic heart rate), "
                "and at what gestational weeks should each appear per clinical guidelines?"
            )
        if st.button("🩺 Gravid vs. Non-Gravid", use_container_width=True, help="Check gestational sac, fetal pole, or non-gravid pelvic landmarks"):
            st.session_state.query_input = (
                "Determine scan status: Is this an obstetric scan showing pregnancy landmarks "
                "(gestational sac, fetus, cardiac activity) or a non-gravid pelvic ultrasound? "
                "Detail all visible anatomical structures."
            )
        if st.button("💧 Placenta & Fluid Volume", use_container_width=True, help="Amniotic fluid index and placental localization"):
            st.session_state.query_input = (
                "What are the standard ultrasound evaluation criteria for amniotic fluid volume "
                "(Single Deepest Pocket vs AFI) and placental location/grading across pregnancy?"
            )

    with preset_cols[1]:
        if st.button("📅 Gestational Age & Dating", use_container_width=True, help="Ultrasound biometry benchmarks for pregnancy dating"):
            st.session_state.query_input = (
                "How is gestational age calculated on ultrasound across trimesters? "
                "Detail Crown-Rump Length (CRL), Biparietal Diameter (BPD), Head Circumference (HC), "
                "Abdominal Circumference (AC), and Femur Length (FL) accuracy per ISUOG practice standards."
            )
        if st.button("🫀 5-Plane Cardiac Screening", use_container_width=True, help="Evaluate 4CH, LVOT, RVOT, 3VT, and abdomen"):
            st.session_state.query_input = (
                "Perform full ISUOG 2023 fetal cardiac screening. "
                "Which standard cardiac planes are visualized (4-Chamber, LVOT, RVOT, 3VT, Stomach/Situs)? "
                "Evaluate axis, symmetry, and outflow tract crossover."
            )
        if st.button("⚠️ PUL & Ectopic Red Flags", use_container_width=True, help="Pregnancy of Unknown Location and emergency signs"):
            st.session_state.query_input = (
                "What are the diagnostic protocols and sonographic criteria for Pregnancy of Unknown Location (PUL) "
                "and ectopic pregnancy? When is serum quantitative beta-hCG correlation required?"
            )

    # ── Clinical Question Input Area ──
    user_query = st.text_area(
        "Pregnancy Support & Ultrasound Consultation Query",
        value=st.session_state.query_input,
        height=105,
        placeholder="e.g. Ask any pregnancy query (e.g. signs of viable pregnancy, CRL vs gestational age, ISUOG scan criteria) or query an uploaded scan...",
        help="Type any pregnancy support inquiry directly, or attach an ultrasound image/sweep for multimodal analysis.",
    )

    # ── Action Buttons ──
    btn_cols = st.columns([2, 1])
    with btn_cols[0]:
        analyze_clicked = st.button("🩺 Submit Pregnancy Query / Analyze", type="primary", use_container_width=True)
    with btn_cols[1]:
        reset_clicked = st.button("🗑️ Reset Case", use_container_width=True)

    if reset_clicked:
        if st.session_state.session_id:
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
        for k in ["session_id", "chat_history", "citations", "frame_count", "is_video", "uploaded_filename", "query_input"]:
            st.session_state[k] = [] if k in ("chat_history", "citations") else \
                                   False if k == "is_video" else \
                                   0 if k == "frame_count" else \
                                   "" if k == "query_input" else None
        st.success("Session reset.")
        st.rerun()

    # ── Execution Logic for /analyze ──
    if analyze_clicked:
        if not user_query.strip():
            st.warning("⚠️ Please provide a pregnancy support query or select a preset.")
        elif not is_online:
            st.error("❌ Cannot run analysis: Backend API server is offline. Check localhost:8000.")
        else:
            with st.status("🔬 Processing Pregnancy Support & Ultrasound Pipeline...", expanded=True) as status_box:
                t_start = time.time()
                if uploaded:
                    status_box.write("1️⃣ Ingesting scan & sampling optical flow keyframes...")
                else:
                    status_box.write("1️⃣ Formulating pregnancy query & consulting clinical parameters...")
                time.sleep(0.1)
                
                status_box.write("2️⃣ Querying ISUOG Guideline Knowledge Base (ChromaDB RAG)...")
                
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)} if uploaded else {"file": ("", b"")}
                    resp = requests.post(
                        f"{API_URL}/analyze",
                        files=files,
                        data={"query": user_query.strip(), "k": k_frames},
                        timeout=140,
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.is_video = data.get("is_video", False)
                        st.session_state.frame_count = data.get("frame_count", 0)
                        st.session_state.citations = data.get("citations", [])
                        
                        now_str = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history = [
                            {"role": "user", "content": user_query.strip(), "timestamp": now_str},
                            {"role": "assistant", "content": data["answer"], "timestamp": now_str},
                        ]
                        
                        elapsed = time.time() - t_start
                        status_box.update(label=f"✅ Response generated in {elapsed:.1f}s!", state="complete", expanded=False)
                        st.rerun()
                    else:
                        status_box.update(label="❌ Query Failed", state="error")
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    status_box.update(label="❌ Connection Timeout or Error", state="error")
                    st.error(f"Failed to communicate with backend: {e}")

    # ── Clinical Assistant Guidance Box ──
    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚖️ Clinical Decision Support Disclaimer</strong><br>
        EchoAI is an assistive AI tool powered by ISUOG practice guidelines. Output is strictly intended for educational and clinical decision support and does not replace certified sonographer/fetal cardiologist diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN: Clinical Report, Interactive Chat & Guideline Grounding
# ══════════════════════════════════════════════════════════════════════════════
with right_col:
    # Diagnostic Session Overview Banner (if active)
    if st.session_state.session_id and st.session_state.chat_history:
        latest_ans = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else ""
        
        # Determine status pill highlight from response text
        if "PREGNANCY DETECTED" in latest_ans or "PREGNANT" in latest_ans:
            st.markdown("""
            <div class="alert-gravid">
                <strong>🤰 SCAN CLASSIFICATION: PREGNANCY DETECTED</strong><br>
                Obstetric fetal / gestational structures identified and grounded in ISUOG guidelines.
            </div>
            """, unsafe_allow_html=True)
        elif "NON-PREGNANT" in latest_ans or "NON-GRAVID" in latest_ans:
            st.markdown("""
            <div class="alert-nongravid">
                <strong>🩺 SCAN CLASSIFICATION: NON-GRAVID SCAN</strong><br>
                Pelvic or abdominal anatomy identified without intrauterine or extrauterine gestational sac.
            </div>
            """, unsafe_allow_html=True)
        elif "INDETERMINATE" in latest_ans:
            st.markdown("""
            <div class="alert-indeterminate">
                <strong>⚠️ SCAN CLASSIFICATION: INDETERMINATE / EARLY CORRELATION REQUIRED</strong><br>
                Equivocal ultrasound landmarks. Clinical correlation with serum β-hCG and repeat TVS advised.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-gravid" style="border-left-color: #38bdf8; background: rgba(56, 189, 248, 0.1);">
                <strong>💡 PREGNANCY SUPPORT & CLINICAL CONSULTATION</strong><br>
                Evidence-based maternal-fetal assessment grounded in ISUOG practice standards.
            </div>
            """, unsafe_allow_html=True)

    # ── Tabs Navigation ──
    tab_chat, tab_report, tab_citations, tab_telemetry = st.tabs([
        "💬 Clinical Consultation",
        "📋 Structured Assessment",
        f"📚 Guideline Citations ({len(st.session_state.citations)})",
        "🔬 Session Telemetry",
    ])

    # ── TAB 1: Interactive Chat ──
    with tab_chat:
        if not st.session_state.chat_history:
            st.info("👋 Enter a pregnancy support query or select a preset on the left (with or without an ultrasound upload), then click **Submit Pregnancy Query / Analyze**.")
        else:
            for turn in st.session_state.chat_history:
                role = turn["role"]
                content = turn["content"]
                timestamp = turn.get("timestamp", "")
                
                if role == "user":
                    with st.chat_message("user", avatar="🤰"):
                        st.markdown(f"**Consulting Clinician / Patient Query** `{timestamp}`")
                        st.markdown(content)
                else:
                    with st.chat_message("assistant", avatar="🩺"):
                        st.markdown(f"**EchoAI Pregnancy Specialist** `{timestamp}`")
                        st.markdown(content)

            # ── Follow-Up Conversation Input ──
            if st.session_state.session_id:
                st.markdown("---")
                follow_up_query = st.chat_input("Ask a follow-up clinical question on this scan …")
                
                if follow_up_query:
                    with st.spinner("Analyzing follow-up in context of prior scan and ISUOG guidelines …"):
                        try:
                            resp = requests.post(
                                f"{API_URL}/chat",
                                data={
                                    "session_id": st.session_state.session_id,
                                    "follow_up": follow_up_query,
                                    "k": k_frames if 'k_frames' in locals() else 8,
                                },
                                timeout=110,
                            )
                            if resp.status_code == 200:
                                res_data = resp.json()
                                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                                st.session_state.chat_history.append({"role": "user", "content": follow_up_query, "timestamp": now_str})
                                st.session_state.chat_history.append({"role": "assistant", "content": res_data["answer"], "timestamp": now_str})
                                if "citations" in res_data and res_data["citations"]:
                                    st.session_state.citations = res_data["citations"]
                                st.rerun()
                            else:
                                st.error(f"Error {resp.status_code}: {resp.text}")
                        except Exception as e:
                            st.error(f"Chat communication error: {e}")

    # ── TAB 2: Structured Assessment & Report Export ──
    with tab_report:
        if st.session_state.chat_history:
            # Extract last assistant report
            assistant_reports = [t["content"] for t in st.session_state.chat_history if t["role"] == "assistant"]
            if assistant_reports:
                latest_report = assistant_reports[-1]
                
                # Report Action Bar
                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    st.download_button(
                        label="📥 Download Clinical Report (.md)",
                        data=latest_report,
                        file_name=f"echoai_report_{st.session_state.session_id[:8] if st.session_state.session_id else 'scan'}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col_d2:
                    st.download_button(
                        label="📄 Export Full Consultation Log (.json)",
                        data=json.dumps(st.session_state.chat_history, indent=2),
                        file_name=f"echoai_consultation_{st.session_state.session_id[:8] if st.session_state.session_id else 'case'}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                
                st.markdown("### 📋 Executive Medical Summary")
                st.markdown(latest_report)
        else:
            st.info("No assessment report generated yet. Ingest an ultrasound file to produce a structured summary.")

    # ── TAB 3: ISUOG Guideline Citations ──
    with tab_citations:
        if st.session_state.citations:
            st.markdown(f"**Verified Evidence**: `{len(st.session_state.citations)}` excerpts retrieved from ISUOG 2023 Guidelines via ChromaDB vector store.")
            
            for c in st.session_state.citations:
                cid = c.get("id", "•")
                text = c.get("text", "")
                
                st.markdown(f"""
                <div class="citation-chip">
                    <div class="citation-header">📖 ISUOG Guideline Excerpt [{cid}]</div>
                    {text}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Guideline citations will be displayed here following analysis.")

    # ── TAB 4: Session Telemetry ──
    with tab_telemetry:
        st.markdown("### 🔬 System & Session Diagnostics")
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Active Session ID", st.session_state.session_id[:8] + "…" if st.session_state.session_id else "None")
        col_t2.metric("Scan Media Type", "Cine Sweep (Video)" if st.session_state.is_video else "Static B-Mode Frame")
        col_t3.metric("Processed Keyframes", st.session_state.frame_count)
        
        st.markdown("#### Active Configuration")
        st.json({
            "API_URL": API_URL,
            "Backend Health": "Online" if st.session_state.backend_online else "Offline",
            "Latency (ms)": st.session_state.last_latency_ms,
            "RAG Vector Store": "ChromaDB (isuog_rag - 118 chunks)",
            "Vision Backbone": "GPT-4o Multimodal Vision (High Detail)",
            "Guidelines Standard": "ISUOG 2023 Practice Guidelines: Fetal Cardiac Screening",
        })


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Clinical Intelligence Hub & Reference Standards
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤰 Pregnancy & Fetal Intelligence Hub")
    
    st.markdown("""
    **EchoAI** supports maternal-fetal consultation and automated ultrasound screening grounded in **ISUOG Practice Standards**:

    **1️⃣ First Trimester (5–12 Weeks)**
    - **4.5–5 wks**: Gestational Sac (GS) within decidua
    - **5–5.5 wks**: Yolk sac confirms intrauterine gestation
    - **6–6.5 wks**: Fetal pole & cardiac flicker (>100 bpm)
    - **7–12 wks**: Crown-Rump Length (CRL) dating accuracy ±3-5 days
    - **11–13+6 wks**: Nuchal Translucency (NT) aneuploidy screening

    **2️⃣ Second Trimester (18–22 Weeks)**
    - Detailed Anatomic Survey & 5-Plane Fetal Echocardiography:
      * Situs & Stomach
      * 4-Chamber View (45° axis)
      * Left Outflow (LVOT)
      * Right Outflow (RVOT)
      * 3 Vessels & Trachea (3VT)
    - Biometry: BPD, Head Circumference, AC, Femur Length

    **3️⃣ Third Trimester (28–40 Weeks)**
    - Growth trajectory & estimated fetal weight
    - Amniotic fluid volume (SDP normal 2–8 cm; AFI 5–25 cm)
    - Placental localization (exclude previa) & umbilical Doppler
    """)
    st.divider()

    st.markdown("### 📊 Active Pipeline Architecture")
    st.markdown("""
    - **Pipeline 1**: Ingestion & Optical Flow Keyframe Extraction
    - **Pipeline 2**: RAG Query Expansion & ChromaDB Vector Store
    - **Pipeline 3**: Multimodal Vision Fusion (GPT-4o) + Clinical Guardrails
    """)
    st.divider()

    st.markdown("### 🩺 Session Controls")
    if st.session_state.session_id:
        st.caption(f"Session UUID: `{st.session_state.session_id}`")
        if st.button("🔄 Clear Active Session", use_container_width=True):
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
            for k in ["session_id", "chat_history", "citations", "frame_count", "is_video", "uploaded_filename", "query_input"]:
                st.session_state[k] = [] if k in ("chat_history", "citations") else \
                                       False if k == "is_video" else \
                                       0 if k == "frame_count" else \
                                       "" if k == "query_input" else None
            st.rerun()

    st.caption("EchoAI Multimodal Medical System · v2.1")

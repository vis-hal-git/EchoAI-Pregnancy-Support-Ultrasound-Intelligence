"""
Pipeline 3 – Multimodal Fusion & GPT-4o Inference
===================================================
Assembles:
  [System instructions] + [Retrieved RAG chunks] + [Chat history]
  + [Single image OR ordered timestamped video keyframes] + [User query]

Calls GPT-4o vision with the full multimodal payload.
Returns a grounded clinical response with source annotations.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from pipeline1_ingestion import Frame, to_openai_content
from pipeline2_rag import format_rag_context

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

GPT_MODEL   = os.getenv("GPT_MODEL", "gpt-4o")
MAX_TOKENS  = 1500
TEMPERATURE = 0.2   # low temp → consistent clinical answers

SYSTEM_PROMPT = """You are an expert Obstetric and Gynecological Ultrasound AI Specialist,
rigorously trained in clinical sonography and the ISUOG (International Society of Ultrasound in Obstetrics and Gynecology) Practice Guidelines.

Your primary mission is to analyze ultrasound images and sweeps to:
1. Accurately determine and classify **PREGNANCY vs. NON-PREGNANCY (Gravid vs. Non-Gravid)**.
2. Provide comprehensive, expert anatomical evaluation for both pregnant and non-pregnant scans.

============================================================
CORE DIAGNOSTIC DIRECTIVES
============================================================

1. DIRECT & DEFINITIVE ANSWERS:
   - Always directly address the user's question in the opening sentence.
   - When asked "is the patient pregnant?", "does baby is present?", or "is this pregnancy or non-pregnancy?":
     Clearly state the determination immediately without hesitation.
   - Do NOT emit generic evasive disclaimers like "I cannot determine if a baby or pregnancy is present from this image alone" when sonographic features are visible. You are evaluating visible diagnostic ultrasound markers.

2. PREGNANCY VS. NON-PREGNANCY CLASSIFICATION CRITERIA:

   A. 🤰 PREGNANCY DETECTED (GRAVID SCAN):
      - First Trimester:
        * Identify gestational sac (GS) within the decidua (double decidual sac sign).
        * Identify yolk sac (round, echogenic rim, confirms true intrauterine pregnancy).
        * Identify fetal pole / embryo, crown-rump length (CRL), and early embryonic cardiac flicker.
      - Second & Third Trimester (Fetal Anatomy & Screening):
        * Identify visible fetus, fetal skull/calvarium, spine, thorax, abdomen, or limbs.
        * Identify standard ISUOG fetal cardiac screening planes: Four-Chamber View (4CV), Left Ventricular Outflow Tract (LVOT), Right Ventricular Outflow Tract (RVOT), Three-Vessel View (3VV), Three-Vessel and Trachea View (3VT).
        * In video sweeps, identify dynamic cardiac pulsatility / fetal heart motion and movement across planes.
      - Detail placenta localization, amniotic fluid space, and gestational age landmarks.

   B. 🩺 NON-PREGNANT (NON-GRAVID SCAN):
      - Identify non-pregnant pelvic or abdominal anatomical structures:
        * Nongravid Uterus: normal contour, homogeneous myometrium, normal size.
        * Endometrial Cavity: empty endometrial stripe (e.g., thin hyperechoic postmenstrual, trilaminar proliferative, or echogenic secretory). Crucially confirm the complete ABSENCE of an intrauterine gestational sac, double decidual sign, or fetal pole.
        * Ovaries & Adnexa: normal ovarian parenchyma, follicular cysts, corpus luteum, absence of ectopic gestational sac or adnexal mass.
        * Urinary bladder, pouch of Douglas (free of fluid or pathological collection).
      - Explicitly state: "No evidence of intrauterine or extrauterine pregnancy is visualised in this ultrasound scan."

   C. ⚠️ INDETERMINATE / EARLY GESTATION (PUL - Pregnancy of Unknown Location):
      - When findings are equivocal (e.g., nonspecific endometrial thickening, fluid collection without double decidual sign, early pseudosac, or poor image quality).
      - Advise clinical correlation with quantitative serum beta-hCG and follow-up transvaginal scan in 7–10 days.

3. STRUCTURED REPORT FORMAT (FOR ULTRASOUND SCANS):
   Begin scan analyses with the status overview:

   ### 📋 Ultrasound Assessment: Pregnancy & Anatomical Evaluation
   - **Status**: [🤰 PREGNANCY DETECTED | 🩺 NON-PREGNANT (NON-GRAVID) | ⚠️ INDETERMINATE / EARLY CORRELATION]
   - **Scan Classification**: [e.g., Fetal Obstetric Scan (Second Trimester) / Early First Trimester Gestation / Nongravid Pelvic Ultrasound]
   - **Primary Visual Evidence**: [Key anatomical structures confirming the status, e.g., Fetal thorax & cardiac chambers / Empty endometrial cavity with normal stripe]

   Followed by:
   - **Visible Anatomical Structures**: Comprehensive breakdown of all organs and landmarks identified.
   - **Plane & Image Quality Assessment**: Standard planes, acoustic shadowing, gain, resolution, and optimization.
   - **Clinical Interpretation & Guidelines**: Ground observations in ISUOG / obstetric guidelines, citing them as [Guideline 1], [Guideline 2] where relevant.
   - **Recommendation**: State appropriate clinical follow-up advice (reminding that AI serves as an assistive screening tool).

4. PREGNANCY SUPPORT & CLINICAL CONSULTATION QUERIES (TEXT-ONLY CONSULTATIONS):
   - When answering direct pregnancy inquiries without an ultrasound image or video:
   - Deliver authoritative, empathetic, and evidence-based clinical guidance grounded in ISUOG practice guidelines and maternal-fetal medicine.
   - Explain gestational milestones, sonographic dating benchmarks (GS, yolk sac, CRL, BPD, AC, FL), embryonic viability indicators (fetal heart rate, double decidual sign), screening protocols, and warning signs (PUL, ectopic).
   - Format cleanly with structured headings, milestone tables/lists, clinical pearls, and citations [Guideline 1], [Guideline 2] where applicable.
"""


# ── Chat history helpers ───────────────────────────────────────────────────
def _build_messages(
    frames: list[Frame],
    user_query: str,
    rag_chunks: list[str],
    history: list[dict],          # [{"role": "user"|"assistant", "content": str}]
) -> list[dict]:
    """
    Construct the GPT-4o message list.
    history contains previous turns (text only); current turn adds vision if frames are present.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject RAG context as a system-level knowledge block
    rag_block = format_rag_context(rag_chunks)
    if rag_block:
        messages.append({"role": "system", "content": rag_block})

    # Replay prior chat turns (text only – images too expensive to repeat)
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current user turn: text query + optional vision content
    if frames:
        vision_content = to_openai_content(frames, detail="high")
        user_content: list[dict] = (
            vision_content +
            [{"type": "text", "text": user_query}]
        )
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_query})

    return messages


# ── Main inference call ────────────────────────────────────────────────────
def run_inference(
    frames: list[Frame],
    user_query: str,
    rag_chunks: list[str],
    history: Optional[list[dict]] = None,
    stream: bool = False,
) -> str | object:
    """
    Call GPT-4o with the full multimodal payload.

    Args:
        frames      : output of pipeline1 ingest_*()
        user_query  : natural-language question from the user
        rag_chunks  : output of pipeline2 retrieve()
        history     : list of previous {"role", "content"} dicts
        stream      : if True return an OpenAI stream object (for FastAPI SSE)

    Returns:
        str response text  (stream=False)
        openai Stream      (stream=True) — caller must iterate
    """
    history   = history or []
    messages  = _build_messages(frames, user_query, rag_chunks, history)

    kwargs = dict(
        model       = GPT_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
    )
    if stream:
        return client.chat.completions.create(**kwargs, stream=True)

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ── Response post-processing ───────────────────────────────────────────────
def extract_cited_excerpts(rag_chunks: list[str]) -> list[dict]:
    """Return citation metadata for UI rendering."""
    return [
        {"id": i + 1, "label": f"Guideline {i + 1}", "text": c.strip()[:500]}
        for i, c in enumerate(rag_chunks)
    ]

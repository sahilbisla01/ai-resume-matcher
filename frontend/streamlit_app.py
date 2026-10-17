"""
streamlit_app.py
-----------------
Interactive dashboard for the Autonomous Job Description & Resume Matcher
Agent. Talks to the FastAPI backend (see backend/app.py) over HTTP.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import textwrap

import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure backend directory is accessible for direct docx generation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
try:
    from docx_export import generate_docx_bytes
except ImportError:
    generate_docx_bytes = None

try:
    from sample_data import SAMPLE_JD_TEXT, SAMPLE_RESUME_TEXT, get_sample_docx_bytes, ensure_samples_dir
    ensure_samples_dir()
except Exception:
    SAMPLE_JD_TEXT = ""
    SAMPLE_RESUME_TEXT = ""
    get_sample_docx_bytes = None
    ensure_samples_dir = None

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # seconds; LLM calls can take a while

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, .main-title {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #C084FC;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFFFFF 25%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
    }
    .hero-title span {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.96rem;
        line-height: 1.55;
        margin: 0;
        max-width: 900px;
    }

    /* Score & Metric Cards */
    .score-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.65) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .score-number {
        font-size: 4.4rem;
        font-weight: 900;
        line-height: 1;
        font-family: 'Outfit', sans-serif;
    }
    .score-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.4rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        margin: 0.25rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .badge:hover {
        transform: translateY(-1px);
    }
    .badge-missing {
        background-color: rgba(239, 68, 68, 0.14);
        color: #FCA5A5 !important;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-matched {
        background-color: rgba(34, 197, 94, 0.14);
        color: #86EFAC !important;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .badge-soft {
        background-color: rgba(245, 158, 11, 0.14);
        color: #FCD34D !important;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    /* Red Flags & Bullets */
    .flag-box {
        background-color: rgba(251, 146, 60, 0.12);
        color: #FED7AA !important;
        border-left: 4px solid #FB923C;
        border-top: 1px solid rgba(251, 146, 60, 0.15);
        border-right: 1px solid rgba(251, 146, 60, 0.15);
        border-bottom: 1px solid rgba(251, 146, 60, 0.15);
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.65rem;
        border-radius: 8px;
        font-size: 0.93rem;
        font-weight: 500;
        line-height: 1.5;
    }
    .bullet-before {
        background-color: rgba(239, 68, 68, 0.12);
        color: #FECACA !important;
        border-left: 4px solid #F87171;
        border-top: 1px solid rgba(239, 68, 68, 0.15);
        border-right: 1px solid rgba(239, 68, 68, 0.15);
        border-bottom: 1px solid rgba(239, 68, 68, 0.15);
        padding: 0.95rem 1.15rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        line-height: 1.6;
        font-size: 0.93rem;
    }
    .bullet-after {
        background-color: rgba(34, 197, 94, 0.12);
        color: #BBF7D0 !important;
        border-left: 4px solid #4ADE80;
        border-top: 1px solid rgba(34, 197, 94, 0.15);
        border-right: 1px solid rgba(34, 197, 94, 0.15);
        border-bottom: 1px solid rgba(34, 197, 94, 0.15);
        padding: 0.95rem 1.15rem;
        border-radius: 8px;
        line-height: 1.6;
        font-weight: 500;
        font-size: 0.93rem;
    }

    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.55);
    }

    /* Step Cards */
    .step-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 0.65rem 0.85rem;
        border-radius: 9px;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
        color: #CBD5E1;
        line-height: 1.4;
    }
    .step-card b {
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "resume_bytes" not in st.session_state:
    st.session_state.resume_bytes = None
if "resume_name" not in st.session_state:
    st.session_state.resume_name = None
if "upgraded_resume" not in st.session_state:
    st.session_state.upgraded_resume = None
if "cover_letter" not in st.session_state:
    st.session_state.cover_letter = None
if "jd_textarea" not in st.session_state:
    st.session_state.jd_textarea = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_badges(items: list[str], css_class: str) -> str:
    if not items:
        return "<em>None</em>"
    return "".join(f'<span class="badge {css_class}">{item}</span>' for item in items)


def score_color(score: int) -> str:
    if score >= 80:
        return "#15803D"  # green
    if score >= 55:
        return "#CA8A04"  # amber
    return "#B91C1C"  # red


def check_backend_health() -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    # Standalone mode on Streamlit Community Cloud
    return {"status": "ok", "mode": "standalone", "available_providers": ["groq"]}


def call_backend(
    endpoint: str,
    resume_bytes: bytes,
    filename: str,
    jd_text: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    # 1. Try remote FastAPI server if reachable
    try:
        content_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if filename.lower().endswith(".docx")
            else "application/pdf"
        )
        files = {"resume": (filename, resume_bytes, content_type)}
        data = {"jd_text": jd_text}
        if provider:
            data["provider"] = provider
        if model:
            data["model"] = model
        if api_key:
            data["api_key"] = api_key
        resp = requests.post(f"{BACKEND_URL}{endpoint}", files=files, data=data, timeout=REQUEST_TIMEOUT)
        if resp.status_code < 400:
            return resp.json()
    except Exception:
        pass

    # 2. In-Process Standalone Fallback (Ideal for Streamlit Community Cloud)
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    try:
        import importlib
        import agents
        import nlp_extractor
        importlib.reload(nlp_extractor)
        importlib.reload(agents)
        from parser import parse_resume_file, clean_jd_text
    except ImportError as e:
        raise RuntimeError(f"Could not reach backend and local modules not found: {e}")

    parsed = parse_resume_file(resume_bytes, filename)
    cleaned_jd = clean_jd_text(jd_text)

    if endpoint == "/cover-letter":
        letter = agents.run_cover_letter_generator(
            parsed.cleaned_text, cleaned_jd, provider=provider, model=model, api_key=api_key
        )
        return {"cover_letter": letter}
    elif endpoint == "/upgrade-resume":
        upgraded = agents.run_full_resume_upgrader(
            parsed.cleaned_text, cleaned_jd, provider=provider, model=model, api_key=api_key
        )
        return {"upgraded_resume": upgraded}
    elif endpoint == "/match":
        entities = nlp_extractor.extract_entities(parsed.cleaned_text)
        baseline = nlp_extractor.compute_baseline_score(parsed.cleaned_text, cleaned_jd)
        res = agents.run_full_pipeline(
            parsed.cleaned_text, cleaned_jd, baseline, provider=provider, model=model, api_key=api_key
        )
        return {
            "parse": {
                "cleaned_text": parsed.cleaned_text,
                "page_count": parsed.page_count,
                "extraction_method": parsed.extraction_method,
                "warnings": parsed.warnings,
                "entities": {
                    "technical_skills": entities.technical_skills,
                    "soft_skills": entities.soft_skills,
                    "degrees": entities.degrees,
                    "years_experience": entities.years_experience,
                },
            },
            "evaluation": {
                "ats_score": res.evaluation.ats_score,
                "score_rationale": res.evaluation.score_rationale,
                "critical_missing_hard_skills": res.evaluation.critical_missing_hard_skills,
                "critical_missing_soft_skills": res.evaluation.critical_missing_soft_skills,
                "matched_strengths": res.evaluation.matched_strengths,
                "formatting_red_flags": res.evaluation.formatting_red_flags,
                "overall_recommendation": res.evaluation.overall_recommendation,
                "baseline": {
                    "overlap_score": baseline.overlap_score,
                    "keyword_match_score": baseline.keyword_match_score,
                    "matched_skills": baseline.matched_skills,
                    "missing_skills": baseline.missing_skills,
                },
            },
            "rewrite": {
                "tailored_bullets": [
                    {
                        "original": b.original,
                        "rewritten": b.rewritten,
                        "jd_keywords_used": b.jd_keywords_used,
                        "improvement_rationale": b.improvement_rationale,
                    }
                    for b in res.tailoring.tailored_bullets
                ]
            },
        }
    raise RuntimeError(f"Unknown endpoint: {endpoint}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.2rem;">
            <span style="font-size:1.8rem;">🎯</span>
            <div>
                <div style="font-size:1.25rem; font-weight:800; font-family:'Outfit',sans-serif; color:#F8FAFC; line-height:1.2;">ResumeAI</div>
                <div style="font-size:0.75rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">ATS Matcher & Optimizer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    health = check_backend_health()
    if health:
        is_direct = health.get("mode") == "standalone"
        status_label = "Cloud Direct (Groq Active)" if is_direct else "Backend Online (Groq Active)"
        st.markdown(
            f"""
            <div style="background:rgba(34, 197, 94, 0.12); border:1px solid rgba(34, 197, 94, 0.35); border-radius:8px; padding:0.55rem 0.8rem; display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
                <span style="height:9px; width:9px; background-color:#22C55E; border-radius:50%; display:inline-block; box-shadow:0 0 8px #22C55E;"></span>
                <span style="color:#86EFAC; font-weight:700; font-size:0.85rem;">{status_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"❌ Cannot reach backend at {BACKEND_URL}")
        st.caption("Start it with: `uvicorn app:app --reload` in backend/")

    st.divider()
    st.subheader("⚡ AI Engine")

    selected_provider = "groq"
    model_options = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ]
    selected_model = st.selectbox(
        "Groq Model",
        options=model_options,
        index=0,
        help="Ultra-fast open-source LLMs hosted on Groq Cloud."
    )
    selected_label = f"Groq ({selected_model})"
    custom_api_key = None

    st.divider()
    st.subheader("💡 How It Works")
    st.markdown(
        """
        <div class="step-card"><b>1. 📄 Upload Resume:</b> PDF or Word (.docx)</div>
        <div class="step-card"><b>2. 🎯 Paste Target JD:</b> Target job description</div>
        <div class="step-card"><b>3. ⚡ Run Analysis:</b> Instant ATS score & gap analysis</div>
        <div class="step-card"><b>4. ✍️ Tailored Bullets:</b> Google X-Y-Z formula rewrites</div>
        <div class="step-card"><b>5. ✉️ Cover Letter:</b> Tailored, job-specific cover letter</div>
        <div class="step-card"><b>6. 🚀 1-Click Upgrade:</b> Download complete new resume in <b>.docx (Word)</b> or <b>.txt</b>!</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-tag">✨ AI-Powered ATS Optimizer & Career Copilot</div>
        <h1 class="hero-title">ResumeAI <span>Matcher & Optimizer</span></h1>
        <p class="hero-subtitle">
            Scan your resume against any job description, uncover critical skill gaps, elevate weak achievements with Google's X-Y-Z formula, and export an ATS 90+ optimized resume in 1-click.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Side-by-side input: Resume upload (left) / JD text (right)
# ---------------------------------------------------------------------------

# Sample / Demo pre-loader bar
demo_col1, demo_col2 = st.columns([2.5, 1])
with demo_col1:
    load_demo_clicked = st.button(
        "✨ Load Sample Resume & Job Description (1-Click Demo)",
        use_container_width=True,
        help="Loads pre-configured Senior Full-Stack Python & AI Engineer profile for instant ATS testing.",
    )
    if load_demo_clicked:
        if get_sample_docx_bytes:
            st.session_state.resume_bytes = get_sample_docx_bytes()
            st.session_state.resume_name = "sample_resume.docx"
        st.session_state.jd_textarea = SAMPLE_JD_TEXT
        st.session_state.match_result = None
        st.session_state.cover_letter = None
        st.session_state.upgraded_resume = None
        st.rerun()

with demo_col2:
    if st.button("🔄 Reset to Blank", use_container_width=True):
        st.session_state.resume_bytes = None
        st.session_state.resume_name = None
        st.session_state.jd_textarea = ""
        st.session_state.match_result = None
        st.session_state.cover_letter = None
        st.session_state.upgraded_resume = None
        st.rerun()

st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.subheader("📄 Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or Word DOCX)",
        type=["pdf", "docx"],
        help="Supports PDF and DOCX files (up to 10MB). Or click 'Load Sample Resume' above.",
    )
    if uploaded_file is not None:
        st.session_state.resume_bytes = uploaded_file.getvalue()
        st.session_state.resume_name = uploaded_file.name
        st.success(f"Loaded: {uploaded_file.name} ({len(st.session_state.resume_bytes) / 1024:.0f} KB)")
    elif st.session_state.resume_bytes:
        st.info(f"📋 **Active Resume:** `{st.session_state.resume_name}` ({len(st.session_state.resume_bytes) / 1024:.0f} KB)")

with right:
    st.subheader("🧾 Job Description")
    jd_text = st.text_area(
        "Paste the target job description",
        value=st.session_state.jd_textarea,
        height=280,
        placeholder="Paste the full job description text here, or click '✨ Load Sample Resume & Job Description' above...",
        key="jd_textarea",
    )

st.divider()

btn_col1, btn_col2, _ = st.columns([2.5, 1, 1.5])
with btn_col1:
    run_clicked = st.button("🚀 Run Instant ATS Match & Optimization", type="primary", use_container_width=True)
with btn_col2:
    clear_clicked = st.button("🗑️ Clear Results", use_container_width=True)

if clear_clicked:
    st.session_state.match_result = None
    st.session_state.cover_letter = None
    st.session_state.upgraded_resume = None
    st.session_state.resume_bytes = None
    st.session_state.resume_name = None
    st.session_state.jd_textarea = ""
    st.rerun()

if run_clicked:
    if not st.session_state.resume_bytes:
        st.error("Please upload a resume (PDF or DOCX) or click '✨ Load Sample Resume & Job Description'.")
    elif not jd_text or not jd_text.strip():
        st.error("Please paste a job description or click '✨ Load Sample Resume & Job Description'.")
    elif not health:
        st.error("Backend is unreachable. Start the FastAPI server and try again.")
    else:
        with st.spinner(f"Analyzing resume using {selected_label}..."):
            try:
                result = call_backend(
                    "/match",
                    st.session_state.resume_bytes,
                    st.session_state.resume_name or "resume.pdf",
                    jd_text,
                    provider=selected_provider,
                    model=selected_model,
                    api_key=custom_api_key.strip() if custom_api_key else None,
                )
                st.session_state.match_result = result
            except Exception as exc:
                st.error(f"Backend error: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.match_result

if result:
    evaluation = result["evaluation"]
    rewrite = result["rewrite"]
    parse = result["parse"]

    st.markdown('<h2 style="font-family:Outfit,sans-serif; margin-bottom:1.2rem;">📊 Match Results & ATS Fit Analysis</h2>', unsafe_allow_html=True)

    score = evaluation["ats_score"]
    color = score_color(score)

    score_col, meta_col = st.columns([1, 2], gap="medium")

    with score_col:
        st.markdown(
            f"""
            <div class="score-container" style="text-align:center;">
                <div class="score-number" style="color:{color}; text-shadow: 0 0 35px {color}40;">{score}</div>
                <div class="score-label">ATS Match Score / 100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(score, 0), 100) / 100)

    with meta_col:
        st.markdown(
            f"""
            <div class="score-container">
                <div style="font-weight:700; font-size:1.05rem; color:#F8FAFC; margin-bottom:0.4rem; font-family:Outfit,sans-serif;">🎯 Score Rationale</div>
                <p style="color:#CBD5E1; font-size:0.95rem; line-height:1.55; margin-bottom:0.8rem;">{evaluation["score_rationale"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        baseline = evaluation["baseline"]
        b1, b2 = st.columns(2)
        b1.metric("Baseline Token Overlap", f"{baseline['overlap_score']:.0f}%")
        b2.metric("Baseline Keyword Match", f"{baseline['keyword_match_score']:.0f}%")

    st.divider()

    # --- Missing / matched keyword badges ---
    st.subheader("🏷️ Keyword Analysis")

    tag_col1, tag_col2, tag_col3 = st.columns(3)
    with tag_col1:
        st.markdown("**❌ Missing Hard Skills**")
        st.markdown(
            render_badges(evaluation["critical_missing_hard_skills"], "badge-missing"),
            unsafe_allow_html=True,
        )
    with tag_col2:
        st.markdown("**❌ Missing Soft Skills**")
        st.markdown(
            render_badges(evaluation["critical_missing_soft_skills"], "badge-soft"),
            unsafe_allow_html=True,
        )
    with tag_col3:
        st.markdown("**✅ Matched Strengths**")
        st.markdown(
            render_badges(evaluation["matched_strengths"], "badge-matched"),
            unsafe_allow_html=True,
        )

    st.caption(
        "💡 Click any badge above visually to spot gaps — copy these keywords "
        "directly into your resume's skills section where truthful."
    )

    st.divider()

    # --- Formatting red flags ---
    st.subheader("🚩 Formatting Red Flags")
    red_flags = evaluation["formatting_red_flags"]
    if red_flags:
        for flag in red_flags:
            st.markdown(f'<div class="flag-box">⚠️ {flag}</div>', unsafe_allow_html=True)
    else:
        st.success("No major formatting red flags detected.")

    st.divider()

    # --- Overall recommendation ---
    st.subheader("🧭 Overall Recommendation")
    st.info(evaluation["overall_recommendation"])

    st.divider()

    # --- Before / After bullet points ---
    st.header("✍️ Bullet Point Tailoring (Before → After)")
    bullets = rewrite["tailored_bullets"]

    if not bullets:
        st.warning("The agent did not identify any bullets to rewrite.")
    else:
        for i, b in enumerate(bullets, start=1):
            with st.expander(f"Bullet {i}: {b['original'][:70]}{'...' if len(b['original']) > 70 else ''}", expanded=(i == 1)):
                col_before, col_after = st.columns(2)
                with col_before:
                    st.markdown("**Before**")
                    st.markdown(f'<div class="bullet-before">{b["original"]}</div>', unsafe_allow_html=True)
                with col_after:
                    st.markdown("**After (X-Y-Z format)**")
                    st.markdown(f'<div class="bullet-after">{b["rewritten"]}</div>', unsafe_allow_html=True)

                st.caption(f"**Why it's stronger:** {b['improvement_rationale']}")
                if b["jd_keywords_used"]:
                    st.markdown(
                        "**JD keywords used:** " + render_badges(b["jd_keywords_used"], "badge-matched"),
                        unsafe_allow_html=True,
                    )

                # Copy-to-clipboard button via a small embedded HTML/JS snippet.
                copy_html = f"""
                <textarea id="bullet-{i}" style="position:absolute; left:-9999px;">{b['rewritten']}</textarea>
                <button onclick="
                    navigator.clipboard.writeText(document.getElementById('bullet-{i}').value);
                    const btn = document.getElementById('copy-btn-{i}');
                    btn.innerText = 'Copied!';
                    setTimeout(() => btn.innerText = '📋 Copy rewritten bullet', 1500);
                " id="copy-btn-{i}" style="
                    background-color:#4ADE80; color:#052e16; border:none;
                    padding:0.4rem 0.9rem; border-radius:6px; cursor:pointer;
                    font-weight:600; margin-top:0.5rem;">
                    📋 Copy rewritten bullet
                </button>
                """
                st.components.v1.html(copy_html, height=50)

    # --- Tailored Cover Letter Generator ---
    st.header("✉️ Tailored Cover Letter Generator")
    if "cover_letter" not in st.session_state:
        st.session_state.cover_letter = None

    if st.button("✨ Generate Custom Cover Letter", use_container_width=False):
        with st.spinner("Generating tailored cover letter based on your resume & JD..."):
            try:
                cl_res = call_backend(
                    "/cover-letter",
                    st.session_state.resume_bytes,
                    st.session_state.resume_name or "resume.pdf",
                    jd_text,
                    provider=selected_provider,
                    model=selected_model,
                    api_key=custom_api_key.strip() if custom_api_key else None,
                )
                st.session_state.cover_letter = cl_res.get("cover_letter", "")
            except Exception as exc:
                st.error(f"Failed to generate cover letter: {exc}")

    if st.session_state.cover_letter:
        st.text_area("Your Tailored Cover Letter", st.session_state.cover_letter, height=260)
        st.download_button(
            label="📥 Download Cover Letter (.txt)",
            data=st.session_state.cover_letter,
            file_name="tailored_cover_letter.txt",
            mime="text/plain",
            use_container_width=False,
        )

    st.divider()

    # --- Full Upgraded Resume Generator ---
    st.header("📄 Full Upgraded ATS-Optimized Resume (1-Click Generator)")
    st.caption(
        "Generate a complete, professionally rewritten resume tailored to this job description. "
        "The AI re-aligns your summary, elevates weak bullets with Google's X-Y-Z formula, "
        "and incorporates critical missing keywords naturally to target a 90+ ATS score."
    )

    if st.button("🚀 Generate Full Upgraded Resume", type="primary", use_container_width=False):
        with st.spinner("Upgrading your complete resume for maximum ATS score (approx. 10-20 seconds)..."):
            try:
                up_res = call_backend(
                    "/upgrade-resume",
                    st.session_state.resume_bytes,
                    st.session_state.resume_name or "resume.pdf",
                    jd_text,
                    provider=selected_provider,
                    model=selected_model,
                    api_key=custom_api_key.strip() if custom_api_key else None,
                )
                st.session_state.upgraded_resume = up_res.get("upgraded_resume", "")
                st.success("🎉 Your upgraded resume has been generated!")
            except Exception as exc:
                st.error(f"Failed to generate upgraded resume: {exc}")

    if st.session_state.upgraded_resume:
        with st.expander("👁️ Preview Upgraded Resume", expanded=True):
            st.markdown(st.session_state.upgraded_resume)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            docx_data = None
            if generate_docx_bytes:
                try:
                    docx_data = generate_docx_bytes(st.session_state.upgraded_resume)
                except Exception:
                    pass
            if docx_data:
                st.download_button(
                    label="📄 Download Upgraded Resume (.docx - Word)",
                    data=docx_data,
                    file_name="Upgraded_Resume_ATS_Optimized.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
        with dl_col2:
            st.download_button(
                label="📝 Download Upgraded Resume (.txt / Markdown)",
                data=st.session_state.upgraded_resume,
                file_name="Upgraded_Resume_ATS_Optimized.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    # --- Downloadable rewritten resume summary ---
    st.header("⬇️ Download Report")

    report_lines = [
        "AI RESUME MATCHER — ANALYSIS REPORT",
        "=" * 40,
        f"ATS Match Score: {score}/100",
        "",
        "Score Rationale:",
        textwrap.fill(evaluation["score_rationale"], 90),
        "",
        "Missing Hard Skills: " + ", ".join(evaluation["critical_missing_hard_skills"] or ["None"]),
        "Missing Soft Skills: " + ", ".join(evaluation["critical_missing_soft_skills"] or ["None"]),
        "Matched Strengths: " + ", ".join(evaluation["matched_strengths"] or ["None"]),
        "",
        "Formatting Red Flags:",
        *[f"  - {f}" for f in (evaluation["formatting_red_flags"] or ["None"])],
        "",
        "Overall Recommendation:",
        textwrap.fill(evaluation["overall_recommendation"], 90),
        "",
        "=" * 40,
        "TAILORED BULLET POINTS (Before -> After)",
        "=" * 40,
    ]
    for i, b in enumerate(bullets, start=1):
        report_lines += [
            f"\n{i}. BEFORE: {b['original']}",
            f"   AFTER:  {b['rewritten']}",
            f"   WHY:    {b['improvement_rationale']}",
        ]

    report_text = "\n".join(report_lines)

    st.download_button(
        label="Download full report (.txt)",
        data=report_text,
        file_name="resume_match_report.txt",
        mime="text/plain",
        use_container_width=False,
    )

    with st.expander("🔍 View extracted resume text & entities (debug)"):
        st.markdown(f"**Extraction method:** `{parse['extraction_method']}` | **Pages:** {parse['page_count']}")
        if parse["warnings"]:
            st.warning("\n".join(parse["warnings"]))
        ent = parse["entities"]
        st.write("**Technical skills detected:**", ", ".join(ent["technical_skills"]) or "None")
        st.write("**Soft skills detected:**", ", ".join(ent["soft_skills"]) or "None")
        st.write("**Degrees detected:**", ", ".join(ent["degrees"]) or "None")
        st.write("**Years of experience mentioned:**", ent["years_experience"] or "None")
        st.text_area("Cleaned resume text", parse["cleaned_text"], height=300)

else:
    st.info("Upload a resume and paste a job description, then click **Run Analysis** to get started.")

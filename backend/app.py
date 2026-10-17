"""
app.py
------
FastAPI server for the Autonomous Job Description & Resume Matcher Agent.

Endpoints:
    GET  /health    - liveness/readiness check
    POST /parse     - upload a resume PDF, get back cleaned text + entities
    POST /evaluate  - upload resume PDF + JD text, run Agent 1 (ATS Evaluator)
    POST /rewrite   - upload resume PDF + JD text, run Agent 2 (Bullet Tailor)
    POST /match     - convenience endpoint: parse + evaluate + rewrite in one call

Run with:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure local imports work regardless of the working directory uvicorn is
# launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agents  # noqa: E402
from docx_export import generate_docx_bytes  # noqa: E402
import nlp_extractor  # noqa: E402
from parser import PDFParsingError, clean_jd_text, parse_resume_file, parse_resume_pdf  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("resume_matcher.app")

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: warming spaCy model...")
    try:
        nlp_extractor.warm_up()
    except RuntimeError as exc:
        logger.error("spaCy warm-up failed: %s", exc)
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")):
        logger.warning(
            "No LLM API keys found. You can set GEMINI_API_KEY (free), "
            "GROQ_API_KEY (free), or ANTHROPIC_API_KEY in .env or pass it in the UI."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Autonomous Job Description & Resume Matcher Agent",
    description=(
        "Parses resumes, extracts entities with spaCy, and runs a "
        "multi-agent LLM pipeline (Gemini, Groq, Claude, Ollama) to score ATS match and rewrite "
        "weak bullet points."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    spacy_loaded: bool
    anthropic_key_configured: bool
    gemini_key_configured: bool
    groq_key_configured: bool
    default_provider: str
    available_providers: list[str]
    model: str


class CoverLetterResponse(BaseModel):
    cover_letter: str


class UpgradedResumeResponse(BaseModel):
    upgraded_resume: str


class EntitiesResponse(BaseModel):
    technical_skills: list[str]
    soft_skills: list[str]
    degrees: list[str]
    years_experience: list[int]


class BaselineResponse(BaseModel):
    overlap_score: float
    keyword_match_score: float
    matched_skills: list[str]
    missing_skills: list[str]


class ParseResponse(BaseModel):
    cleaned_text: str
    page_count: int
    extraction_method: str
    warnings: list[str]
    entities: EntitiesResponse


class EvaluateResponse(BaseModel):
    ats_score: int
    score_rationale: str
    critical_missing_hard_skills: list[str]
    critical_missing_soft_skills: list[str]
    matched_strengths: list[str]
    formatting_red_flags: list[str]
    overall_recommendation: str
    baseline: BaselineResponse


class TailoredBulletResponse(BaseModel):
    original: str
    rewritten: str
    jd_keywords_used: list[str]
    improvement_rationale: str


class RewriteResponse(BaseModel):
    tailored_bullets: list[TailoredBulletResponse]


class MatchResponse(BaseModel):
    parse: ParseResponse
    evaluation: EvaluateResponse
    rewrite: RewriteResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _read_and_validate_resume(resume: UploadFile) -> bytes:
    filename = (resume.filename or "").lower()
    allowed_types = (
        "application/pdf",
        "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    )
    if resume.content_type not in allowed_types and not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a PDF or DOCX file, got content-type '{resume.content_type}'.",
        )
    file_bytes = await resume.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_MB}MB upload limit.",
        )
    return file_bytes


def _entities_to_response(entities: nlp_extractor.ExtractedEntities) -> EntitiesResponse:
    return EntitiesResponse(
        technical_skills=entities.technical_skills,
        soft_skills=entities.soft_skills,
        degrees=entities.degrees,
        years_experience=entities.years_experience,
    )


def _baseline_to_response(baseline: nlp_extractor.BaselineScore) -> BaselineResponse:
    return BaselineResponse(
        overlap_score=baseline.overlap_score,
        keyword_match_score=baseline.keyword_match_score,
        matched_skills=baseline.matched_skills,
        missing_skills=baseline.missing_skills,
    )


def _evaluation_to_response(
    evaluation: agents.ATSEvaluation, baseline: nlp_extractor.BaselineScore
) -> EvaluateResponse:
    return EvaluateResponse(
        ats_score=evaluation.ats_score,
        score_rationale=evaluation.score_rationale,
        critical_missing_hard_skills=evaluation.critical_missing_hard_skills,
        critical_missing_soft_skills=evaluation.critical_missing_soft_skills,
        matched_strengths=evaluation.matched_strengths,
        formatting_red_flags=evaluation.formatting_red_flags,
        overall_recommendation=evaluation.overall_recommendation,
        baseline=_baseline_to_response(baseline),
    )


def _rewrite_to_response(tailoring: agents.BulletTailoringResult) -> RewriteResponse:
    return RewriteResponse(
        tailored_bullets=[
            TailoredBulletResponse(
                original=b.original,
                rewritten=b.rewritten,
                jd_keywords_used=b.jd_keywords_used,
                improvement_rationale=b.improvement_rationale,
            )
            for b in tailoring.tailored_bullets
        ]
    )


def _parse_resume_or_400(file_bytes: bytes, filename: str = "") -> "parser.ParsedResume":
    try:
        return parse_resume_file(file_bytes, filename)
    except PDFParsingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


def _agent_error_or_502(exc: agents.AgentError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness/readiness probe. Reports whether dependencies are configured."""
    spacy_ok = True
    try:
        nlp_extractor._load_nlp()
    except RuntimeError:
        spacy_ok = False

    providers = agents.get_configured_providers()
    def_p, def_m, _ = agents.resolve_provider_and_model()

    return HealthResponse(
        status="ok",
        spacy_loaded=spacy_ok,
        anthropic_key_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
        gemini_key_configured=bool(os.getenv("GEMINI_API_KEY")),
        groq_key_configured=bool(os.getenv("GROQ_API_KEY")),
        default_provider=def_p,
        available_providers=providers,
        model=def_m,
    )


@app.post("/parse", response_model=ParseResponse, tags=["pipeline"])
async def parse_endpoint(resume: UploadFile = File(...)) -> ParseResponse:
    """Extract and clean resume text (PDF or DOCX), then run spaCy entity extraction."""
    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    entities = nlp_extractor.extract_entities(parsed.cleaned_text)

    return ParseResponse(
        cleaned_text=parsed.cleaned_text,
        page_count=parsed.page_count,
        extraction_method=parsed.extraction_method,
        warnings=parsed.warnings,
        entities=_entities_to_response(entities),
    )


@app.post("/baseline", response_model=BaselineResponse, tags=["pipeline"])
async def baseline_endpoint(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
) -> BaselineResponse:
    """
    FREE, no-LLM endpoint: parse the resume + JD and compute the
    deterministic spaCy token/keyword overlap score only.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="jd_text must not be empty.")

    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    cleaned_jd = clean_jd_text(jd_text)

    baseline = nlp_extractor.compute_baseline_score(parsed.cleaned_text, cleaned_jd)
    return _baseline_to_response(baseline)


@app.post("/evaluate", response_model=EvaluateResponse, tags=["pipeline"])
async def evaluate_endpoint(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    api_key: str | None = Form(None),
) -> EvaluateResponse:
    """Run Agent 1 (ATS Evaluator): score resume vs. JD, list gaps and red flags."""
    if not jd_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="jd_text must not be empty.")

    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    cleaned_jd = clean_jd_text(jd_text)

    baseline = nlp_extractor.compute_baseline_score(parsed.cleaned_text, cleaned_jd)

    try:
        evaluation = agents.run_ats_evaluator(
            parsed.cleaned_text, cleaned_jd, baseline, provider=provider, model=model, api_key=api_key
        )
    except agents.AgentError as exc:
        raise _agent_error_or_502(exc) from exc

    return _evaluation_to_response(evaluation, baseline)


@app.post("/rewrite", response_model=RewriteResponse, tags=["pipeline"])
async def rewrite_endpoint(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    api_key: str | None = Form(None),
) -> RewriteResponse:
    """Run Agent 2 (Bullet Tailor): rewrite weak bullets in X-Y-Z format."""
    if not jd_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="jd_text must not be empty.")

    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    cleaned_jd = clean_jd_text(jd_text)

    try:
        tailoring = agents.run_bullet_tailor(
            parsed.cleaned_text, cleaned_jd, provider=provider, model=model, api_key=api_key
        )
    except agents.AgentError as exc:
        raise _agent_error_or_502(exc) from exc

    return _rewrite_to_response(tailoring)


@app.post("/cover-letter", response_model=CoverLetterResponse, tags=["pipeline"])
async def cover_letter_endpoint(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    api_key: str | None = Form(None),
) -> CoverLetterResponse:
    """Generate a tailored, professional cover letter based on resume and job description."""
    if not jd_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="jd_text must not be empty.")

    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    cleaned_jd = clean_jd_text(jd_text)

    try:
        letter = agents.run_cover_letter_generator(
            parsed.cleaned_text, cleaned_jd, provider=provider, model=model, api_key=api_key
        )
    except agents.AgentError as exc:
        raise _agent_error_or_502(exc) from exc

    return CoverLetterResponse(cover_letter=letter)


@app.post("/upgrade-resume", response_model=UpgradedResumeResponse, tags=["pipeline"])
async def upgrade_resume_endpoint(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    api_key: str | None = Form(None),
) -> UpgradedResumeResponse:
    """Agent 4: generate a full, upgraded, ATS-optimized version of the resume."""
    if not jd_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="jd_text must not be empty.")

    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    cleaned_jd = clean_jd_text(jd_text)

    try:
        upgraded = agents.run_full_resume_upgrader(
            parsed.cleaned_text, cleaned_jd, provider=provider, model=model, api_key=api_key
        )
    except agents.AgentError as exc:
        raise _agent_error_or_502(exc) from exc

    return UpgradedResumeResponse(upgraded_resume=upgraded)


@app.post("/export-docx", tags=["pipeline"])
async def export_docx_endpoint(
    content: str = Form(...),
    filename: str = Form("upgraded_resume.docx"),
):
    """Convert text/markdown into a downloadable Microsoft Word (.docx) document."""
    docx_bytes = generate_docx_bytes(content)
    clean_name = filename if filename.endswith(".docx") else f"{filename}.docx"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{clean_name}"'},
    )



@app.post("/match", response_model=MatchResponse, tags=["pipeline"])
async def match_endpoint(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    api_key: str | None = Form(None),
) -> MatchResponse:
    """Convenience endpoint: parse + Agent 1 + Agent 2 in a single call."""
    if not jd_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="jd_text must not be empty.")

    file_bytes = await _read_and_validate_resume(resume)
    parsed = _parse_resume_or_400(file_bytes, resume.filename or "")
    cleaned_jd = clean_jd_text(jd_text)

    entities = nlp_extractor.extract_entities(parsed.cleaned_text)
    baseline = nlp_extractor.compute_baseline_score(parsed.cleaned_text, cleaned_jd)

    try:
        result = agents.run_full_pipeline(
            parsed.cleaned_text, cleaned_jd, baseline, provider=provider, model=model, api_key=api_key
        )
    except agents.AgentError as exc:
        raise _agent_error_or_502(exc) from exc

    return MatchResponse(
        parse=ParseResponse(
            cleaned_text=parsed.cleaned_text,
            page_count=parsed.page_count,
            extraction_method=parsed.extraction_method,
            warnings=parsed.warnings,
            entities=_entities_to_response(entities),
        ),
        evaluation=_evaluation_to_response(result.evaluation, baseline),
        rewrite=_rewrite_to_response(result.tailoring),
    )


@app.get("/sample")
async def get_sample():
    """Returns sample resume and job description text for quick testing."""
    from sample_data import SAMPLE_JD_TEXT, SAMPLE_RESUME_TEXT
    return {
        "job_description": SAMPLE_JD_TEXT,
        "resume_text": SAMPLE_RESUME_TEXT,
        "filename": "sample_resume.docx",
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=True,
    )

"""
agents.py
---------
Multi-agent logic powered by the Anthropic Claude API.

Two agents, each with a narrow, single-responsibility system prompt:

  Agent 1 — ATS Evaluator
      Compares resume text against a job description, returns a strict
      JSON payload with an ATS match score, missing hard/soft skills,
      and formatting red flags.

  Agent 2 — Bullet Point Tailor
      Takes 2-3 weak resume bullet points and rewrites them in Google's
      X-Y-Z format ("Accomplished [X], as measured by [Y], by doing [Z]"),
      tailored to the target JD's keywords.

Both agents:
  - Use `anthropic.Anthropic().messages.create(...)`.
  - Request strict JSON back and are parsed defensively (the LLM
    occasionally wraps JSON in markdown fences despite instructions).
  - Are retried with exponential backoff on transient API errors.
  - Are given the deterministic spaCy baseline score as grounding context
    to reduce hallucinated numbers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from anthropic import Anthropic, APIError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nlp_extractor import BaselineScore

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL

MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_RETRYABLE_EXCEPTIONS = (RateLimitError, APIStatusError, APIError, TimeoutError, httpx.RequestError)


class AgentError(Exception):
    """Raised when an agent fails to produce a usable, parsed result."""


def get_configured_providers() -> list[str]:
    """Return a list of providers that have keys configured or are available."""
    providers = []
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    if os.getenv("GROQ_API_KEY"):
        providers.append("groq")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("claude")
    providers.append("ollama")
    return providers


def resolve_provider_and_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve target provider, model name, and API key."""
    # Auto-detect provider if user supplied a specific key type
    if api_key:
        if api_key.startswith("gsk_"):
            provider = "groq"
        elif api_key.startswith("sk-ant-"):
            provider = "claude"

    p = (provider or os.getenv("LLM_PROVIDER") or "").lower().strip()
    if not p or p == "auto":
        if os.getenv("GROQ_API_KEY"):
            p = "groq"
        elif os.getenv("GEMINI_API_KEY"):
            p = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            p = "claude"
        else:
            p = "groq"

    if p == "groq":
        m = model if (model and "llama" in model.lower()) else DEFAULT_GROQ_MODEL
        k = api_key or os.getenv("GROQ_API_KEY")
    elif p == "gemini":
        m = model if (model and "gemini" in model.lower()) else DEFAULT_GEMINI_MODEL
        k = api_key or os.getenv("GEMINI_API_KEY")
    elif p in ("claude", "anthropic"):
        p = "claude"
        m = model if (model and "claude" in model.lower()) else DEFAULT_ANTHROPIC_MODEL
        k = api_key or os.getenv("ANTHROPIC_API_KEY")
    elif p == "ollama":
        p = "ollama"
        m = model or DEFAULT_OLLAMA_MODEL
        k = None
    else:
        p = "groq"
        m = DEFAULT_GROQ_MODEL
        k = api_key or os.getenv("GROQ_API_KEY")

    return p, m, k



def _get_supported_gemini_models(api_key: str) -> list[str]:
    """Fetch available generateContent models from Gemini API."""
    urls = [
        "https://generativelanguage.googleapis.com/v1beta/models",
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
    ]
    headers = {"x-goog-api-key": api_key}
    for url in urls:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    found = []
                    for m in data.get("models", []):
                        methods = m.get("supportedGenerationMethods", [])
                        if "generateContent" in methods:
                            name = m.get("name", "").removeprefix("models/").strip()
                            if name:
                                found.append(name)
                    if found:
                        return found
        except Exception as exc:
            logger.warning("Could not query Gemini ListModels on %s: %s", url, exc)
    return []


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None,
    max_tokens: int | None = None,
) -> str:
    if not api_key:
        raise AgentError(
            "GEMINI_API_KEY is not set. Get a 100% FREE key from "
            "https://aistudio.google.com/app/apikey and add it to .env or enter it in the sidebar."
        )

    requested_model = (model or DEFAULT_GEMINI_MODEL).removeprefix("models/").strip()

    # Prioritize active models discovered on user's key
    candidate_models = []
    for m in ("gemini-2.5-flash", "gemini-2.5-pro", requested_model, "gemini-2.0-flash", "gemini-1.5-flash"):
        if m and m not in candidate_models:
            candidate_models.append(m)

    gen_config: dict[str, Any] = {
        "temperature": 0.2,
        "maxOutputTokens": max_tokens or MAX_TOKENS,
    }
    if "json" in system_prompt.lower() or "json" in user_prompt.lower():
        gen_config["response_mime_type"] = "application/json"

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": gen_config,
    }

    last_error = ""

    with httpx.Client(timeout=60.0) as client:
        for attempt_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{attempt_model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}

            try:
                resp = client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                raise AgentError(f"Failed to connect to Google Gemini API: {exc}") from exc

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise AgentError("Gemini returned no candidates.")
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise AgentError("Gemini returned no text content.")
                return parts[0]["text"]

            # If 400 Bad Request, retry without response_mime_type in case of model schema variation
            if resp.status_code == 400:
                fallback_payload = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                }
                try:
                    resp_retry = client.post(url, headers=headers, json=fallback_payload)
                    if resp_retry.status_code == 200:
                        parts = resp_retry.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                except Exception:
                    pass

            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err_msg = resp.text

            last_error = f"{attempt_model} ({resp.status_code}): {err_msg}"
            logger.info("Attempt on Gemini model '%s' resulted in: %s", attempt_model, last_error)

    available = _get_supported_gemini_models(api_key)
    avail_msg = f" Available models for your key: {', '.join(available[:5])}" if available else ""
    raise AgentError(
        f"Gemini API Error: {last_error}.{avail_msg}"
    )


def _get_supported_groq_models(api_key: str) -> list[str]:
    """Fetch active models from Groq API for this account."""
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                models = [item.get("id", "") for item in data.get("data", []) if item.get("id")]
                priority_order = [
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-70b-versatile",
                    "llama3-70b-8192",
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                ]
                res = [m for m in priority_order if m in models]
                for m in models:
                    if m not in res and not any(skip in m for skip in ("whisper", "guard", "embed")):
                        res.append(m)
                if res:
                    return res
    except Exception as exc:
        logger.warning("Could not query Groq models list: %s", exc)
    return ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None,
    max_tokens: int | None = None,
) -> str:
    if not api_key:
        raise AgentError(
            "GROQ_API_KEY is not set. Get a FREE key from "
            "https://console.groq.com/keys and add it to .env or enter it in the sidebar."
        )

    requested_model = (model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
    available_models = _get_supported_groq_models(api_key)

    candidate_models: list[str] = []
    if requested_model:
        candidate_models.append(requested_model)
    for m in available_models:
        if m not in candidate_models:
            candidate_models.append(m)
    for fallback in ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for attempt_model in candidate_models:
        payload: dict[str, Any] = {
            "model": attempt_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens or MAX_TOKENS,
        }
        if "json" in system_prompt.lower() or "json" in user_prompt.lower():
            payload["response_format"] = {"type": "json_object"}

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            last_error = f"Connection error: {exc}"
            continue

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")

        try:
            err_msg = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err_msg = resp.text

        last_error = f"{attempt_model} ({resp.status_code}): {err_msg}"
        logger.info("Attempt on Groq model '%s' resulted in: %s", attempt_model, last_error)

        # If it's an unauthorized key error, stop early
        if resp.status_code in (401, 403) and "model" not in err_msg.lower():
            break

    raise AgentError(f"Groq API Error: {last_error}")



def _call_ollama(system_prompt: str, user_prompt: str, model: str, base_url: str | None = None) -> str:
    url = f"{(base_url or OLLAMA_BASE_URL).rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise AgentError(
            f"Failed to connect to local Ollama at {url}. Make sure Ollama is installed and running (`ollama serve`): {exc}"
        ) from exc

    if resp.is_error:
        raise AgentError(f"Ollama Error ({resp.status_code}): {resp.text}")

    data = resp.json()
    return data.get("response", "")


def _call_claude(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int | None = None,
) -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise AgentError(
            "ANTHROPIC_API_KEY is not set. Add your key to .env or switch to FREE "
            "Google Gemini / Groq in the sidebar."
        )

    # Normalize model names to valid official Anthropic model IDs
    target_model = (model or DEFAULT_ANTHROPIC_MODEL).strip()
    if "4-5" in target_model or "sonnet" in target_model:
        target_model = "claude-3-5-sonnet-20241022"
    elif "haiku" in target_model:
        target_model = "claude-3-5-haiku-20241022"
    elif "opus" in target_model:
        target_model = "claude-3-opus-20240229"

    try:
        client = Anthropic(api_key=key)
        response = client.messages.create(
            model=target_model,
            max_tokens=max_tokens or MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except APIStatusError as exc:
        raise AgentError(f"Anthropic Claude API Error ({exc.status_code}): {exc.message}") from exc
    except APIError as exc:
        raise AgentError(f"Anthropic Claude API Error: {exc.message}") from exc
    except Exception as exc:
        raise AgentError(f"Anthropic Claude API Error: {exc}") from exc

    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        raise AgentError("Claude returned no text content.")
    return "\n".join(text_blocks)


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
) -> str:
    p, m, k = resolve_provider_and_model(provider, model, api_key)
    if p == "gemini":
        return _call_gemini(system_prompt, user_prompt, m, k, max_tokens=max_tokens)
    elif p == "groq":
        return _call_groq(system_prompt, user_prompt, m, k, max_tokens=max_tokens)
    elif p == "ollama":
        return _call_ollama(system_prompt, user_prompt, m)
    elif p == "claude":
        return _call_claude(system_prompt, user_prompt, m, k, max_tokens=max_tokens)
    else:
        raise AgentError(f"Unknown LLM provider: {p}")


def _extract_json(raw_text: str) -> dict[str, Any]:
    """
    Defensively parse JSON out of an LLM response, tolerating:
      - Clean JSON
      - JSON wrapped in ```json ... ``` markdown fences
      - Leading/trailing prose the model added despite instructions
    """
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find the outermost { ... } span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise AgentError(f"Could not parse JSON from model response: {exc}") from exc

    raise AgentError("Model response did not contain any JSON.")


# ---------------------------------------------------------------------------
# Agent 1 — ATS Evaluator
# ---------------------------------------------------------------------------

ATS_EVALUATOR_SYSTEM_PROMPT = """\
You are an expert ATS (Applicant Tracking System) evaluator and technical \
recruiter with 15 years of experience screening resumes for top tech \
companies. You compare a candidate's resume against a target job \
description with the rigor of a real ATS parser combined with a human \
recruiter's judgment.

Rules:
- Respond with STRICT, VALID JSON ONLY. No markdown fences, no prose \
before or after the JSON, no trailing commas.
- Be specific and evidence-based: cite exact skills/keywords from the JD \
that are missing, don't invent generic feedback.
- The ats_score must be an integer between 0 and 100, reflecting realistic \
ATS keyword-match + human-recruiter-quality standards (a resume with zero \
relevant keywords should score near 0; a near-perfect tailored match \
should score 90+).
- "critical_missing_hard_skills" and "critical_missing_soft_skills" should \
each contain the JD-required skills the resume does not evidence, ranked \
most-critical first. Omit skills already present in the resume.
- "formatting_red_flags" should flag ATS-parsing risks you can infer from \
the raw extracted text (e.g. dense paragraphs instead of bullets, missing \
section headers, tables/columns that likely broke text extraction, no \
quantified achievements, inconsistent date formats, missing contact info, \
overly long resume). If none are evident, return an empty list.

Return exactly this JSON schema:
{
  "ats_score": <integer 0-100>,
  "score_rationale": "<2-3 sentence explanation of the score>",
  "critical_missing_hard_skills": ["<skill>", ...],
  "critical_missing_soft_skills": ["<skill>", ...],
  "matched_strengths": ["<skill or qualification the resume already covers well>", ...],
  "formatting_red_flags": ["<flag>", ...],
  "overall_recommendation": "<one paragraph of actionable next-step advice>"
}
"""

ATS_EVALUATOR_USER_TEMPLATE = """\
### JOB DESCRIPTION
{jd_text}

### RESUME (extracted text)
{resume_text}

### DETERMINISTIC BASELINE SIGNAL (from a spaCy keyword-matching pass; \
use as a sanity-check reference, not ground truth — your judgment should \
be more nuanced than raw keyword overlap)
- Token overlap score: {overlap_score}/100
- Keyword match score: {keyword_match_score}/100
- Skills matched by keyword pass: {matched_skills}
- Skills the keyword pass flagged as missing: {missing_skills}

Evaluate this resume against the job description now. Return JSON only.
"""


@dataclass
class ATSEvaluation:
    ats_score: int
    score_rationale: str
    critical_missing_hard_skills: list[str] = field(default_factory=list)
    critical_missing_soft_skills: list[str] = field(default_factory=list)
    matched_strengths: list[str] = field(default_factory=list)
    formatting_red_flags: list[str] = field(default_factory=list)
    overall_recommendation: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ATSEvaluation":
        score = data.get("ats_score", 0)
        try:
            score = max(0, min(100, int(score)))
        except (TypeError, ValueError):
            score = 0
        return cls(
            ats_score=score,
            score_rationale=str(data.get("score_rationale", "")),
            critical_missing_hard_skills=list(data.get("critical_missing_hard_skills", []) or []),
            critical_missing_soft_skills=list(data.get("critical_missing_soft_skills", []) or []),
            matched_strengths=list(data.get("matched_strengths", []) or []),
            formatting_red_flags=list(data.get("formatting_red_flags", []) or []),
            overall_recommendation=str(data.get("overall_recommendation", "")),
            raw_response=data,
        )


def run_ats_evaluator(
    resume_text: str,
    jd_text: str,
    baseline: BaselineScore,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ATSEvaluation:
    """Agent 1: score the resume against the JD and flag gaps/red flags."""
    user_prompt = ATS_EVALUATOR_USER_TEMPLATE.format(
        jd_text=jd_text.strip(),
        resume_text=resume_text.strip(),
        overlap_score=baseline.overlap_score,
        keyword_match_score=baseline.keyword_match_score,
        matched_skills=", ".join(baseline.matched_skills) or "none",
        missing_skills=", ".join(baseline.missing_skills) or "none",
    )
    raw = _call_llm(ATS_EVALUATOR_SYSTEM_PROMPT, user_prompt, provider=provider, model=model, api_key=api_key)
    data = _extract_json(raw)
    return ATSEvaluation.from_dict(data)


# ---------------------------------------------------------------------------
# Agent 2 — Bullet Point Tailor
# ---------------------------------------------------------------------------

BULLET_TAILOR_SYSTEM_PROMPT = """\
You are an expert resume writer who specializes in Google's "X-Y-Z" bullet \
formula: "Accomplished [X], as measured by [Y], by doing [Z]." You rewrite \
weak, vague, or under-quantified resume bullet points into sharp, \
achievement-oriented bullets tailored to a specific target job description.

Rules:
- Respond with STRICT, VALID JSON ONLY. No markdown fences, no prose \
before or after the JSON.
- Select the 2-3 WEAKEST bullet points from the resume (vague duties, no \
metrics, passive language, generic responsibilities) — do not rewrite \
bullets that are already strong.
- Every rewritten bullet MUST follow X-Y-Z structure and should naturally \
incorporate 1-3 keywords/skills from the job description where truthful \
and relevant. Never fabricate specific numbers/metrics that were not \
implied by the original bullet — if no metric is available, use a \
reasonable placeholder in square brackets (e.g. "[X%]", "[N users]") so \
the candidate knows to fill in a real number, rather than inventing one.
- Keep each rewritten bullet to a single sentence, resume-appropriate \
length (roughly 20-35 words).
- Explain briefly why each rewrite is stronger.

Return exactly this JSON schema:
{
  "tailored_bullets": [
    {
      "original": "<verbatim original bullet text>",
      "rewritten": "<X-Y-Z formatted rewrite>",
      "jd_keywords_used": ["<keyword>", ...],
      "improvement_rationale": "<1-2 sentences on why this is stronger>"
    }
  ]
}
"""

BULLET_TAILOR_USER_TEMPLATE = """\
### JOB DESCRIPTION
{jd_text}

### RESUME (extracted text)
{resume_text}

Identify the 2-3 weakest bullet points in this resume and rewrite them in \
X-Y-Z format, tailored to the job description above. Return JSON only.
"""


@dataclass
class TailoredBullet:
    original: str
    rewritten: str
    jd_keywords_used: list[str] = field(default_factory=list)
    improvement_rationale: str = ""


@dataclass
class BulletTailoringResult:
    tailored_bullets: list[TailoredBullet] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BulletTailoringResult":
        bullets = []
        for item in data.get("tailored_bullets", []) or []:
            bullets.append(
                TailoredBullet(
                    original=str(item.get("original", "")),
                    rewritten=str(item.get("rewritten", "")),
                    jd_keywords_used=list(item.get("jd_keywords_used", []) or []),
                    improvement_rationale=str(item.get("improvement_rationale", "")),
                )
            )
        return cls(tailored_bullets=bullets)


def run_bullet_tailor(
    resume_text: str,
    jd_text: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> BulletTailoringResult:
    """Agent 2: rewrite the weakest 2-3 bullets into X-Y-Z format."""
    user_prompt = BULLET_TAILOR_USER_TEMPLATE.format(
        jd_text=jd_text.strip(),
        resume_text=resume_text.strip(),
    )
    raw = _call_llm(BULLET_TAILOR_SYSTEM_PROMPT, user_prompt, provider=provider, model=model, api_key=api_key)
    data = _extract_json(raw)
    return BulletTailoringResult.from_dict(data)


# ---------------------------------------------------------------------------
# Agent 3 — Tailored Cover Letter Generator
# ---------------------------------------------------------------------------

COVER_LETTER_SYSTEM_PROMPT = """\
You are an expert career coach and executive recruiter. You write compelling, \
tailored, and professional cover letters that highlight a candidate's genuine \
strengths and align directly with the target job description.

Rules:
- Keep the tone confident, professional, and authentic (no cliché fluff).
- Explicitly connect 2-3 key accomplishments from the candidate's resume with \
  specific requirements in the job description.
- Keep the letter to 3-4 concise paragraphs (approx 250-350 words).
- Provide bracketed placeholders like [Hiring Manager Name or Team] where information \
  is not provided.
- Return ONLY the cover letter text, no extraneous markdown wrappers.
"""


def run_cover_letter_generator(
    resume_text: str,
    jd_text: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Agent 3: generate a personalized, tailored cover letter."""
    user_prompt = (
        f"### TARGET JOB DESCRIPTION:\n{jd_text.strip()}\n\n"
        f"### CANDIDATE RESUME:\n{resume_text.strip()}\n\n"
        "Draft a highly tailored cover letter now."
    )
    return _call_llm(COVER_LETTER_SYSTEM_PROMPT, user_prompt, provider=provider, model=model, api_key=api_key)


# ---------------------------------------------------------------------------
# Agent 4 — Full Resume Upgrader (ATS 90+ Optimizer)
# ---------------------------------------------------------------------------

FULL_RESUME_UPGRADER_SYSTEM_PROMPT = """\
You are an elite Silicon Valley Executive Resume Strategist and ATS (Applicant Tracking System) Optimization Master.
Your mission is to completely upgrade, rewrite, and elevate the candidate's existing resume to achieve maximum ATS compatibility and interview conversion for the target Job Description.

STRICT GUIDELINES:
1. Truthfulness & Identity:
   - Preserve the candidate's real personal identity (Name, Email, Phone, LinkedIn, GitHub, College/University, Degree name, Graduation year, GPA/CGPA). Do not invent fake degrees or fake company names.
2. Target Alignment & Professional Summary:
   - Craft a punchy, high-impact 3-4 sentence Professional Summary tailored directly to the target role. Highlight key technical domains, passion, and value proposition.
3. Structured Skills Section:
   - Organize skills into clean categories:
     * Programming Languages
     * Frameworks & Libraries
     * AI/ML, NLP & Deep Learning (or role-relevant domains)
     * Developer Tools & Databases
     * Core Concepts & Methodologies
   - Organically include the relevant missing technical keywords from the Job Description that the candidate can legitimately claim or has adjacent experience with.
4. Experience & Projects Enhancement (Google X-Y-Z Method):
   - Rewrite project and experience bullet points using Google's formula: "Accomplished [X] as measured by [Y] by doing [Z]".
   - Infuse realistic, compelling metrics (e.g., latency reduction, accuracy/F1-score improvements, data scale, throughput, efficiency).
   - Start every single bullet point with a powerful action verb (Architected, Engineered, Developed, Spearheaded, Optimized, Deployed).
5. ATS-Friendly Formatting:
   - Use clean, standard markdown formatting.
   - Main sections in ALL CAPS:
     # CANDIDATE NAME
     Contact Info (Email | Phone | LinkedIn | GitHub | Location)
     ## PROFESSIONAL SUMMARY
     ## TECHNICAL SKILLS
     ## EDUCATION
     ## PROJECTS
     ## EXPERIENCE (if present)
     ## CERTIFICATIONS & ACHIEVEMENTS (if present)
   - Use standard `- ` for bullet points.
   - DO NOT use tables, columns, or emojis that break ATS parsing.
   - Return ONLY the upgraded resume text in markdown format. No conversational filler or introductory remarks.
"""


def run_full_resume_upgrader(
    resume_text: str,
    jd_text: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Agent 4: rewrite and upgrade the entire resume for the target JD."""
    user_prompt = (
        f"### TARGET JOB DESCRIPTION:\n{jd_text.strip()}\n\n"
        f"### CANDIDATE CURRENT RESUME:\n{resume_text.strip()}\n\n"
        "Generate the complete, upgraded, ATS-optimized resume now."
    )
    return _call_llm(
        FULL_RESUME_UPGRADER_SYSTEM_PROMPT,
        user_prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=4096,
    )



# ---------------------------------------------------------------------------
# Orchestration helper — runs both agents and assembles a combined result
# ---------------------------------------------------------------------------


@dataclass
class MatcherResult:
    evaluation: ATSEvaluation
    tailoring: BulletTailoringResult
    baseline: BaselineScore


def run_full_pipeline(
    resume_text: str,
    jd_text: str,
    baseline: BaselineScore,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> MatcherResult:
    """Run Agent 1 then Agent 2 sequentially and return a combined result."""
    evaluation = run_ats_evaluator(
        resume_text, jd_text, baseline, provider=provider, model=model, api_key=api_key
    )
    tailoring = run_bullet_tailor(
        resume_text, jd_text, provider=provider, model=model, api_key=api_key
    )
    return MatcherResult(evaluation=evaluation, tailoring=tailoring, baseline=baseline)

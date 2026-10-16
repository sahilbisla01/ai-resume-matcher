"""
nlp_extractor.py
-----------------
spaCy-based entity extraction and a cheap, deterministic baseline match
score, computed *before* anything is sent to the LLM.

Why compute a baseline score at all, if Claude will score it anyway?
  1. It's free and instant -> good for showing the user immediate feedback.
  2. It gives Agent 1 (ATS Evaluator) a grounded starting point/reference,
     which we pass into the prompt to reduce hallucinated scores.
  3. It works even if the Anthropic API is unavailable (graceful degradation).

Extraction is intentionally rule-based + spaCy-matcher driven rather than
relying on spaCy's default NER labels, because `en_core_web_sm`'s NER was
trained on OntoNotes and does not natively recognize "skills" as an entity
type. We combine:
  - PhraseMatcher over curated technical/soft-skill vocabularies
  - Regex for years-of-experience patterns
  - PhraseMatcher + regex for degree/education mentions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated vocabularies. In production these would live in a config file or
# database and be regularly refreshed; kept inline here for portability.
# ---------------------------------------------------------------------------

TECHNICAL_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "react", "angular", "vue", "next.js", "node.js", "django", "flask",
    "fastapi", "spring boot", "graphql", "rest api", "microservices",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd",
    "aws", "azure", "gcp", "google cloud", "amazon web services",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pytorch", "tensorflow", "scikit-learn", "keras",
    "pandas", "numpy", "spark", "hadoop", "airflow", "kafka", "etl",
    "data engineering", "data science", "data analysis", "power bi",
    "tableau", "excel", "git", "github", "gitlab", "linux", "bash",
    "agile", "scrum", "jira", "confluence", "html", "css", "sass",
    "webpack", "figma", "product management", "a/b testing", "llm",
    "large language models", "generative ai", "prompt engineering",
    "anthropic", "openai", "hugging face", "vector database", "rag",
    "retrieval augmented generation", "spacy", "streamlit",
]

SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "collaboration",
    "problem solving", "problem-solving", "critical thinking",
    "time management", "adaptability", "creativity", "attention to detail",
    "project management", "stakeholder management", "mentoring",
    "public speaking", "negotiation", "conflict resolution",
    "decision making", "decision-making", "emotional intelligence",
    "cross-functional collaboration", "strategic thinking", "ownership",
    "initiative", "customer focus", "analytical thinking",
]

DEGREE_TERMS = [
    "bachelor of science", "bachelor of arts", "bachelor's degree",
    "master of science", "master of arts", "master's degree",
    "mba", "phd", "ph.d.", "doctorate", "associate degree",
    "b.sc", "b.s.", "b.a.", "m.sc", "m.s.", "m.a.", "b.tech", "m.tech",
    "computer science", "data science", "information technology",
    "electrical engineering", "software engineering",
]

_YEARS_EXPERIENCE_RE = re.compile(
    r"(\d{1,2})\+?\s*(?:-\s*\d{1,2}\s*)?years?\s+(?:of\s+)?experience",
    re.IGNORECASE,
)


@dataclass
class ExtractedEntities:
    technical_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    degrees: list[str] = field(default_factory=list)
    years_experience: list[int] = field(default_factory=list)

    @property
    def max_years_experience(self) -> int | None:
        return max(self.years_experience) if self.years_experience else None


@dataclass
class BaselineScore:
    """A deterministic, non-LLM keyword/token-overlap score."""

    overlap_score: float           # 0-100, Jaccard-style token overlap
    keyword_match_score: float     # 0-100, weighted skill-keyword overlap
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_nlp():
    """Load the spaCy pipeline once and cache it for the process lifetime."""
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception:
        logger.info("Attempting to load or download %s model...", model_name)
        try:
            import spacy.cli as spacy_cli
            spacy_cli.download(model_name)
            return spacy.load(model_name)
        except Exception as exc:
            logger.warning("Could not load %s (%s). Falling back to spacy.blank('en')", model_name, exc)
            return spacy.blank("en")


def _build_matcher(nlp, vocabulary: list[str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in vocabulary]
    matcher.add("VOCAB", patterns)
    return matcher


def _match_phrases(nlp, doc, vocabulary: list[str]) -> list[str]:
    matcher = _build_matcher(nlp, vocabulary)
    matches = matcher(doc)
    found = {doc[start:end].text.lower() for _, start, end in matches}
    # Preserve canonical casing from the source vocabulary where possible.
    canonical = {term.lower(): term for term in vocabulary}
    return sorted({canonical.get(f, f) for f in found})


def extract_entities(text: str) -> ExtractedEntities:
    """Run the spaCy pipeline and extract skills, degrees, and experience."""
    nlp = _load_nlp()
    doc = nlp(text)

    technical = _match_phrases(nlp, doc, TECHNICAL_SKILLS)
    soft = _match_phrases(nlp, doc, SOFT_SKILLS)
    degrees = _match_phrases(nlp, doc, DEGREE_TERMS)

    years = [int(m.group(1)) for m in _YEARS_EXPERIENCE_RE.finditer(text)]

    return ExtractedEntities(
        technical_skills=technical,
        soft_skills=soft,
        degrees=degrees,
        years_experience=years,
    )


def _tokenize_for_overlap(nlp, text: str) -> set[str]:
    doc = nlp(text.lower())
    return {
        (tok.lemma_ if tok.lemma_ else tok.text.lower())
        for tok in doc
        if not tok.is_stop and not tok.is_punct and not tok.is_space and len(tok.text) > 2
    }


def compute_baseline_score(resume_text: str, jd_text: str) -> BaselineScore:
    """
    Compute a fast, deterministic baseline score combining:
      - Jaccard token overlap (lemmatized, stopwords removed) across the
        full resume vs. JD text.
      - Weighted overlap of extracted technical + soft skill keywords,
        which is a stronger signal for ATS-style matching than raw tokens.
    """
    nlp = _load_nlp()

    resume_tokens = _tokenize_for_overlap(nlp, resume_text)
    jd_tokens = _tokenize_for_overlap(nlp, jd_text)

    if jd_tokens:
        intersection = resume_tokens & jd_tokens
        union = resume_tokens | jd_tokens
        overlap_score = round(100 * len(intersection) / max(len(union), 1), 2)
    else:
        overlap_score = 0.0

    resume_entities = extract_entities(resume_text)
    jd_entities = extract_entities(jd_text)

    resume_skills = {s.lower() for s in resume_entities.technical_skills + resume_entities.soft_skills}
    jd_skills = {s.lower() for s in jd_entities.technical_skills + jd_entities.soft_skills}

    if jd_skills:
        matched = jd_skills & resume_skills
        missing = jd_skills - resume_skills
        keyword_match_score = round(100 * len(matched) / len(jd_skills), 2)
    else:
        matched, missing = set(), set()
        keyword_match_score = 0.0

    return BaselineScore(
        overlap_score=overlap_score,
        keyword_match_score=keyword_match_score,
        matched_skills=sorted(matched),
        missing_skills=sorted(missing),
    )


def warm_up() -> None:
    """Force-load the spaCy model at startup so the first request isn't slow."""
    _load_nlp()
    logger.info("spaCy model loaded and ready.")

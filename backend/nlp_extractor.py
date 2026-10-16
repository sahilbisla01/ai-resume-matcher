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

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception:
    spacy = None
    PhraseMatcher = None

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


_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def _match_terms_regex(text: str, vocabulary: list[str]) -> list[str]:
    text_lower = f" {text.lower()} "
    found = []
    for term in vocabulary:
        pattern = r"(?:^|[\s,.;:!?/()\[\]{}])" + re.escape(term.lower()) + r"(?:$|[\s,.;:!?/()\[\]{}])"
        if re.search(pattern, text_lower):
            found.append(term)
    return sorted(set(found))


@lru_cache(maxsize=1)
def _load_nlp():
    """Load the spaCy pipeline once and cache it for the process lifetime."""
    if spacy is None:
        return None
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception as exc:
        logger.warning("Could not load %s (%s). Falling back to blank model or regex.", model_name, exc)
        try:
            return spacy.blank("en")
        except Exception:
            return None


def _build_matcher(nlp, vocabulary: list[str]):
    if PhraseMatcher is None:
        return None
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in vocabulary]
    matcher.add("VOCAB", patterns)
    return matcher


def _match_phrases(nlp, doc, vocabulary: list[str]) -> list[str]:
    matcher = _build_matcher(nlp, vocabulary)
    if matcher is None:
        return []
    matches = matcher(doc)
    found = {doc[start:end].text.lower() for _, start, end in matches}
    canonical = {term.lower(): term for term in vocabulary}
    return sorted({canonical.get(f, f) for f in found})


def extract_entities(text: str) -> ExtractedEntities:
    """Run entity extraction with automatic pure-Python regex fallback."""
    years = [int(m.group(1)) for m in _YEARS_EXPERIENCE_RE.finditer(text)]
    try:
        nlp = _load_nlp()
        if nlp is not None and PhraseMatcher is not None:
            doc = nlp(text)
            technical = _match_phrases(nlp, doc, TECHNICAL_SKILLS)
            soft = _match_phrases(nlp, doc, SOFT_SKILLS)
            degrees = _match_phrases(nlp, doc, DEGREE_TERMS)
            return ExtractedEntities(
                technical_skills=technical,
                soft_skills=soft,
                degrees=degrees,
                years_experience=years,
            )
    except Exception as exc:
        logger.warning("spaCy extraction exception: %s. Using regex fallback.", exc)

    return ExtractedEntities(
        technical_skills=_match_terms_regex(text, TECHNICAL_SKILLS),
        soft_skills=_match_terms_regex(text, SOFT_SKILLS),
        degrees=_match_terms_regex(text, DEGREE_TERMS),
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
      - Jaccard token overlap across full resume vs. JD text.
      - Weighted overlap of extracted technical + soft skill keywords.
    """
    resume_tokens: set[str] = set()
    jd_tokens: set[str] = set()
    try:
        nlp = _load_nlp()
        if nlp is not None:
            resume_tokens = _tokenize_for_overlap(nlp, resume_text)
            jd_tokens = _tokenize_for_overlap(nlp, jd_text)
    except Exception:
        pass

    if not resume_tokens or not jd_tokens:
        resume_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", resume_text.lower())) - _STOPWORDS
        jd_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", jd_text.lower())) - _STOPWORDS

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
    """Force-load the spaCy model at startup."""
    try:
        _load_nlp()
    except Exception:
        pass

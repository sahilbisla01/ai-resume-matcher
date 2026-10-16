"""
parser.py
---------
PDF text extraction for resumes.

Strategy:
    1. Try `pypdf` first (fast, pure-Python, no external deps).
    2. If `pypdf` returns suspiciously little text (common with resumes that
       use multi-column layouts, text boxes, or scanned/embedded fonts),
       fall back to `pdfplumber`, which uses a more thorough layout engine.
    3. Clean the extracted text: strip non-ASCII noise, collapse excessive
       whitespace/newlines, and repair common PDF-extraction glitches
       (e.g. hyphenated line-wraps, bullet glyphs, ligature artifacts).

This module has no knowledge of NLP or LLMs — it only turns bytes into
clean, plain-text strings so downstream modules can work with reliable input.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# A resume shorter than this (in characters) after extraction is treated as
# a signal that pypdf failed to get real content, triggering the fallback.
MIN_ACCEPTABLE_CHARS = 120

# Common ligatures / glyphs that PDF extractors mangle.
_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2022": "- ",   # bullet
    "\u25cf": "- ",   # bullet (solid circle)
    "\u25aa": "- ",   # bullet (small square)
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",    # non-breaking space
}

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")


@dataclass
class ParsedResume:
    """Container for a parsed resume."""

    raw_text: str
    cleaned_text: str
    page_count: int
    extraction_method: str
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.cleaned_text)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed by any available backend."""


def _extract_with_pypdf(file_bytes: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("pypdf failed to extract a page: %s", exc)
            pages_text.append("")
    return "\n".join(pages_text), len(reader.pages)


def _extract_with_pdfplumber(file_bytes: bytes) -> tuple[str, int]:
    import pdfplumber  # imported lazily; heavier dependency

    pages_text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("pdfplumber failed to extract a page: %s", exc)
                pages_text.append("")
    return "\n".join(pages_text), page_count


def _normalize_ligatures(text: str) -> str:
    for bad, good in _LIGATURE_MAP.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    """
    Clean raw PDF-extracted text:
      - Normalize unicode (NFKC) and ligatures.
      - Strip stray non-ASCII/non-printable characters.
      - Repair hyphenated line-wraps ("develop-\nment" -> "development").
      - Collapse excessive blank lines and repeated whitespace.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _normalize_ligatures(text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _NON_PRINTABLE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line != "")

    return text.strip()


def parse_resume_pdf(file_bytes: bytes) -> ParsedResume:
    """
    Extract and clean text from a resume PDF.

    Tries pypdf first; falls back to pdfplumber if pypdf yields too little
    text or throws. Raises PDFParsingError if both backends fail.
    """
    warnings: list[str] = []

    # --- Attempt 1: pypdf ---
    try:
        raw_text, page_count = _extract_with_pypdf(file_bytes)
        method = "pypdf"
    except (PdfReadError, Exception) as exc:  # noqa: BLE001
        logger.warning("pypdf raised an exception, will try pdfplumber: %s", exc)
        raw_text, page_count, method = "", 0, "pypdf_failed"
        warnings.append(f"pypdf extraction failed: {exc}")

    cleaned = clean_text(raw_text)

    # --- Fallback: pdfplumber ---
    if len(cleaned) < MIN_ACCEPTABLE_CHARS:
        warnings.append(
            f"pypdf produced only {len(cleaned)} usable characters; "
            "falling back to pdfplumber."
        )
        try:
            raw_text_pb, page_count_pb = _extract_with_pdfplumber(file_bytes)
            cleaned_pb = clean_text(raw_text_pb)
            if len(cleaned_pb) > len(cleaned):
                raw_text, cleaned, page_count = raw_text_pb, cleaned_pb, page_count_pb
                method = "pdfplumber"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pdfplumber extraction also failed: {exc}")

    if len(cleaned) < MIN_ACCEPTABLE_CHARS:
        raise PDFParsingError(
            "Could not extract readable text from this PDF using pypdf or "
            "pdfplumber. The file may be a scanned image without OCR text, "
            "password-protected, or corrupted."
        )

    return ParsedResume(
        raw_text=raw_text,
        cleaned_text=cleaned,
        page_count=page_count,
        extraction_method=method,
        warnings=warnings,
    )


def _extract_with_docx(file_bytes: bytes) -> tuple[str, int]:
    """Extract text paragraphs from a Word document (.docx) using built-in XML parsing."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for elem in tree.iter():
                if elem.tag.endswith("p"):
                    p_text = "".join(node.text for node in elem.iter() if node.tag.endswith("t") and node.text)
                    if p_text.strip():
                        paragraphs.append(p_text.strip())
            return "\n".join(paragraphs), 1
    except Exception as exc:
        raise PDFParsingError(f"Failed to parse DOCX file: {exc}") from exc


def parse_resume_file(file_bytes: bytes, filename: str = "") -> ParsedResume:
    """
    Extract and clean text from a resume file (supports PDF and DOCX).
    """
    if filename.lower().endswith(".docx"):
        raw_text, page_count = _extract_with_docx(file_bytes)
        cleaned = clean_text(raw_text)
        if len(cleaned) < MIN_ACCEPTABLE_CHARS:
            raise PDFParsingError("DOCX file contains too little readable text.")
        return ParsedResume(
            raw_text=raw_text,
            cleaned_text=cleaned,
            page_count=page_count,
            extraction_method="docx",
            warnings=[],
        )
    return parse_resume_pdf(file_bytes)


def clean_jd_text(text: str) -> str:
    """Apply the same cleaning pass to a pasted job-description string."""
    return clean_text(text)

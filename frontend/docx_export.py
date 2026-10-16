"""
docx_export.py
--------------
Pure Python Microsoft Word (.docx) generator for frontend.
Creates standard, valid .docx files without requiring external pip dependencies.
Uses Python's standard library `zipfile` and XML formatting.
"""

from __future__ import annotations

import io
import re
import xml.sax.saxutils as saxutils
import zipfile

CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def generate_docx_bytes(markdown_or_plain_text: str) -> bytes:
    """
    Converts plain text or simple markdown resume into a valid .docx binary buffer.
    Recognizes headers, bold text, bullets, and standard paragraphs.
    """
    lines = markdown_or_plain_text.splitlines()
    body_xml_parts: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            body_xml_parts.append("<w:p/>")
            continue

        # Check for heading (# Heading or ALL CAPS SECTION HEADER)
        is_h1 = line.startswith("# ")
        is_h2 = line.startswith("## ")
        is_h3 = line.startswith("### ")
        is_caps_header = (
            len(line) > 2
            and line.isupper()
            and any(c.isalpha() for c in line)
            and len(line.split()) <= 4
        )

        if is_h1 or is_h2 or is_h3 or is_caps_header:
            clean_title = re.sub(r"^#+\s*", "", line)
            escaped = saxutils.escape(clean_title)
            size_val = "32" if is_h1 else ("28" if (is_h2 or is_caps_header) else "24")
            p_xml = (
                f'<w:p>'
                f'<w:pPr><w:spacing w:before="180" w:after="80"/></w:pPr>'
                f'<w:r>'
                f'<w:rPr><w:b/><w:sz w:val="{size_val}"/><w:szCs w:val="{size_val}"/><w:color w:val="1E293B"/></w:rPr>'
                f'<w:t xml:space="preserve">{escaped}</w:t>'
                f'</w:r>'
                f'</w:p>'
            )
            body_xml_parts.append(p_xml)
            continue

        # Check for bullet point
        is_bullet = line.startswith(("- ", "* ", "• ", "+ "))
        if is_bullet:
            bullet_text = re.sub(r"^[-*•+]\s*", "", line)
            escaped = saxutils.escape(bullet_text)
            # Handle inline bold like **text**
            escaped = _format_inline_bold(escaped)
            p_xml = (
                f'<w:p>'
                f'<w:pPr><w:ind w:left="400"/><w:spacing w:after="60"/></w:pPr>'
                f'<w:r><w:t xml:space="preserve">•  </w:t></w:r>'
                f'{escaped}'
                f'</w:p>'
            )
            body_xml_parts.append(p_xml)
            continue

        # Standard paragraph
        escaped = saxutils.escape(line)
        formatted_runs = _format_inline_bold(escaped)
        p_xml = (
            f'<w:p>'
            f'<w:pPr><w:spacing w:after="80"/></w:pPr>'
            f'{formatted_runs}'
            f'</w:p>'
        )
        body_xml_parts.append(p_xml)

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        '  <w:body>\n'
        + "\n".join(body_xml_parts)
        + '\n    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
        + '\n  </w:body>\n</w:document>'
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML.encode("utf-8"))
        zf.writestr("_rels/.rels", RELS_XML.encode("utf-8"))
        zf.writestr("word/document.xml", document_xml.encode("utf-8"))

    return buf.getvalue()


def _format_inline_bold(escaped_text: str) -> str:
    """Parses simple **bold** markup into OpenXML w:r chunks."""
    parts = re.split(r"(\*\*.*?\*\*)", escaped_text)
    runs = []
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**") and len(p) >= 4:
            clean = p[2:-2]
            runs.append(
                f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{clean}</w:t></w:r>'
            )
        else:
            runs.append(
                f'<w:r><w:t xml:space="preserve">{p}</w:t></w:r>'
            )
    return "".join(runs)

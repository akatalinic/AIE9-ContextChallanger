import logging
from io import BytesIO

import pdfplumber
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)


def extract_text(filename: str, content_type: str, content: bytes) -> str:
    logger.info("Extraction started | filename=%s content_type=%s size_bytes=%s", filename, content_type, len(content))
    lowered = filename.lower()
    if content_type == "application/pdf" or lowered.endswith(".pdf"):
        text = _extract_pdf(content)
        logger.info("PDF extraction finished | filename=%s text_chars=%s", filename, len(text))
        return text
    if (
        content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or lowered.endswith(".docx")
    ):
        text = _extract_docx(content)
        logger.info("DOCX extraction finished | filename=%s text_chars=%s", filename, len(text))
        return text
    logger.warning("Extraction rejected unsupported format | filename=%s content_type=%s", filename, content_type)
    raise ValueError("Unsupported file type. Only PDF and DOCX are allowed.")


def _extract_pdf(content: bytes) -> str:
    with pdfplumber.open(BytesIO(content)) as pdf:
        parts = [(page.extract_text() or "").strip() for page in pdf.pages]
    return "\n\n".join(p for p in parts if p).strip()


def _extract_docx(content: bytes) -> str:
    doc = Document(BytesIO(content))
    parts: list[str] = []
    for block in _iter_docx_blocks(doc):
        text = block.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _iter_docx_blocks(doc: DocxDocument):
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            if text:
                yield text
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            table_lines = _extract_table_lines(table)
            if table_lines:
                yield "\n".join(table_lines)


def _extract_table_lines(table: Table) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            cell_text = " ".join(
                paragraph.text.strip() for paragraph in cell.paragraphs if paragraph.text.strip()
            ).strip()
            cells.append(cell_text)
        cleaned = [value for value in cells if value]
        if not cleaned:
            continue
        # Keep table data readable and chunk-friendly.
        line = " | ".join(cleaned)
        if line:
            lines.append(line)
    return lines

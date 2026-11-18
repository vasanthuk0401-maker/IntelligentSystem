"""CV parser & document tools: turn an uploaded CV (PDF / DOCX / TXT) into text."""
from __future__ import annotations

import io
from pathlib import Path


def parse_cv_bytes(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts).strip()
    if ext in {".txt", ".md", ""}:
        return data.decode("utf-8", errors="ignore").strip()
    raise ValueError(f"Unsupported CV format: {ext} (use PDF, DOCX or TXT)")


def parse_cv_file(path: str | Path) -> str:
    p = Path(path)
    return parse_cv_bytes(p.read_bytes(), p.name)

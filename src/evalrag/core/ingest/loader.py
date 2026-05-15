import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pypdf
from pypdf.errors import PdfReadError

log = logging.getLogger(__name__)


class LoaderError(Exception):
    pass


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def load(path: str | Path) -> Document:
    p = Path(path)
    if not p.is_file():
        raise LoaderError(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        raise LoaderError(f"unsupported file type: {p.suffix}")
    if suffix in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        text = _load_pdf(p)
    else:  # .docx
        text = _load_docx(p)
    text = text.strip()
    if not text:
        raise LoaderError("empty document")
    return Document(text=text, metadata={"filename": p.name, "ext": suffix})


def _load_pdf(p: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(p))
        if reader.is_encrypted:
            raise LoaderError("encrypted pdf not supported")
    except LoaderError:
        raise
    except (PdfReadError, ValueError, OSError) as e:
        log.warning("pdf encryption check failed on %s, continuing: %s", p.name, e)

    # pdfplumber with a tighter x_tolerance preserves spaces better for
    # resume-style PDFs with dense text, links, and inline bold spans.
    try:
        import pdfplumber

        with pdfplumber.open(str(p)) as pdf:
            text = "\n".join(
                (page.extract_text(x_tolerance=2) or "") for page in pdf.pages
            )
        if text.strip():
            return text
    except Exception as e:
        log.warning("pdfplumber failed on %s, trying pypdf: %s", p.name, e)

    try:
        reader = pypdf.PdfReader(str(p))
        parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(parts)
        if text.strip():
            return text
    except (PdfReadError, ValueError, OSError) as e:
        raise LoaderError(f"pdf parse failed: {e}") from e
    raise LoaderError("pdf parse failed: no extractable text")


def _load_docx(p: Path) -> str:
    from docx import Document as Docx

    d = Docx(str(p))
    return "\n".join(par.text for par in d.paragraphs)

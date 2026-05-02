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
        parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(parts)
        if text.strip():
            return text
    except LoaderError:
        raise
    except (PdfReadError, ValueError, OSError) as e:
        log.warning("pypdf failed on %s, trying pdfplumber: %s", p.name, e)
    # fallback: pdfplumber (covers cases where pypdf returns empty or raises a parse error)
    try:
        import pdfplumber

        with pdfplumber.open(str(p)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as e:
        raise LoaderError(f"pdf parse failed: {e}") from e


def _load_docx(p: Path) -> str:
    from docx import Document as Docx

    d = Docx(str(p))
    return "\n".join(par.text for par in d.paragraphs)

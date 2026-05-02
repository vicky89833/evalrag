import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LIDoc

from evalrag.config import get_settings
from evalrag.core.ingest.loader import Document

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


class ChunkerError(Exception):
    pass


@dataclass
class Chunk:
    """In-memory domain chunk. Distinct from `evalrag.storage.models.Chunk` (DB row).

    `parent_id` is set only for the markdown branch (one parent per heading
    section). The plain-sentence branch leaves it None — there is no hierarchy
    to track. Downstream code must not assume `parent_id` is always present.
    """
    chunk_id: str
    text: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _derive_section(chunk_text: str, fallback: str) -> str:
    """Pull the leading heading out of the chunk text for stratified sampling
    in Task 5.1. Falls back to the LlamaIndex header_path or '_default'."""
    m = _HEADING_RE.search(chunk_text)
    if m:
        return m.group(1).strip()
    return fallback


def chunk(doc: Document) -> list[Chunk]:
    if not doc.text.strip():
        raise ChunkerError("empty document")
    s = get_settings()
    use_markdown = doc.metadata.get("ext") == ".md" and "#" in doc.text
    li_doc = LIDoc(text=doc.text, metadata=doc.metadata)
    splitter = SentenceSplitter(chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP)

    if use_markdown:
        nodes = MarkdownNodeParser().get_nodes_from_documents([li_doc])
        out: list[Chunk] = []
        for n in nodes:
            header_path = n.metadata.get("header_path", "/")
            for piece in splitter.split_text(n.get_content()):
                meta = dict(n.metadata)
                meta["section"] = _derive_section(piece, fallback=header_path)
                out.append(Chunk(chunk_id=uuid4().hex[:16], text=piece,
                                 parent_id=n.node_id, metadata=meta))
        return out

    out_plain: list[Chunk] = []
    for piece in splitter.split_text(doc.text):
        meta = dict(doc.metadata)
        meta["section"] = "_default"
        out_plain.append(Chunk(chunk_id=uuid4().hex[:16], text=piece, metadata=meta))
    return out_plain

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LIDoc

from evalrag.config import get_settings
from evalrag.core.ingest.loader import Document


class ChunkerError(Exception):
    pass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk(doc: Document) -> list[Chunk]:
    if not doc.text.strip():
        raise ChunkerError("empty document")
    s = get_settings()
    use_markdown = doc.metadata.get("ext") == ".md" and "#" in doc.text
    li_doc = LIDoc(text=doc.text, metadata=doc.metadata)
    if use_markdown:
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents([li_doc])
        # If MD parser returns oversized nodes, split them further
        splitter = SentenceSplitter(chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP)
        out: list[Chunk] = []
        for n in nodes:
            sub = splitter.split_text(n.get_content())
            for t in sub:
                out.append(Chunk(chunk_id=uuid4().hex[:16], text=t,
                                 parent_id=n.node_id, metadata=dict(n.metadata)))
        return out
    splitter = SentenceSplitter(chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP)
    pieces = splitter.split_text(doc.text)
    return [Chunk(chunk_id=uuid4().hex[:16], text=t, metadata=dict(doc.metadata)) for t in pieces]

from dataclasses import dataclass


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str  # "dense" | "sparse" | "rerank"

from dataclasses import dataclass
from typing import Literal

HitSource = Literal["dense", "sparse", "rerank", "fused"]


@dataclass(frozen=True, slots=True)
class Hit:
    """Single retrieval result. Score semantics depend on `source`:

    - dense: cosine similarity in [0, 2] (typically [0, 1]); higher is better
    - sparse: ts_rank, unbounded non-negative; higher is better
    - rerank: cross-encoder logit, unbounded; higher is better
    - fused: RRF score, sum of 1/(k+rank); higher is better
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: HitSource

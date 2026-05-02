from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from evalrag.core.ingest.embedder import Embedder
from evalrag.core.retrieval import Hit
from evalrag.core.retrieval.bm25_index import BM25Index
from evalrag.core.retrieval.vector_store import VectorStore


def rrf_fuse(rankings: Sequence[list[Hit]], k: int = 60, top: int = 20) -> list[Hit]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.chunk_id] += 1.0 / (k + rank + 1)
            by_id.setdefault(hit.chunk_id, hit)
    fused = sorted(by_id.values(), key=lambda h: scores[h.chunk_id], reverse=True)[:top]
    return [replace(h, score=scores[h.chunk_id], source="fused") for h in fused]


class Retriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Index, embedder: Embedder) -> None:
        self.vs = vector_store
        self.bm = bm25
        self.emb = embedder

    def retrieve(self, query: str, k: int, doc_id: str) -> list[Hit]:
        qvec = self.emb.embed([query])[0]
        dense = self.vs.search(qvec, k=k, doc_id=doc_id)
        sparse = self.bm.search(query, k=k, doc_id=doc_id)
        if not dense and not sparse:
            return []
        if not sparse:
            return dense[:k]
        if not dense:
            return sparse[:k]
        return rrf_fuse([dense, sparse], top=k)

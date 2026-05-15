from dataclasses import replace
from math import exp

from sentence_transformers import CrossEncoder

from evalrag.config import get_settings
from evalrag.core.retrieval import Hit


class Reranker:
    def __init__(self) -> None:
        self._model = CrossEncoder(get_settings().RERANK_MODEL)

    def rerank(self, query: str, hits: list[Hit], top: int) -> list[Hit]:
        if not hits:
            return []
        raw_scores = self._model.predict([(query, h.text) for h in hits])
        scored = [(h, 1.0 / (1.0 + exp(-float(s)))) for h, s in zip(hits, raw_scores, strict=True)]
        ranked = sorted(scored, key=lambda p: p[1], reverse=True)
        return [replace(h, score=s, source="rerank") for h, s in ranked[:top]]

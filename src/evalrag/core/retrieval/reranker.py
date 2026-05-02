from dataclasses import replace

from sentence_transformers import CrossEncoder

from evalrag.config import get_settings
from evalrag.core.retrieval import Hit


class Reranker:
    def __init__(self) -> None:
        self._model = CrossEncoder(get_settings().RERANK_MODEL)

    def rerank(self, query: str, hits: list[Hit], top: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda p: float(p[1]), reverse=True)
        return [replace(h, score=float(s), source="rerank") for h, s in ranked[:top]]

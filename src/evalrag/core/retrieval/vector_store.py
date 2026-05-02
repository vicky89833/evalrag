import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalrag.core.retrieval import Hit
from evalrag.storage.models import Chunk


class VectorStore:
    def __init__(self, session: Session) -> None:
        self.s = session

    def search(self, query_vec: np.ndarray, k: int, doc_id: str) -> list[Hit]:
        # cosine distance via pgvector "<=>"; score = 1 - distance
        stmt = (
            select(Chunk, Chunk.embedding.cosine_distance(query_vec.tolist()).label("dist"))
            .where(Chunk.doc_id == doc_id)
            .order_by("dist")
            .limit(k)
        )
        rows = self.s.execute(stmt).all()
        return [Hit(chunk_id=c.chunk_id, doc_id=str(c.doc_id), text=c.text,
                    score=float(1.0 - dist), source="dense")
                for c, dist in rows]

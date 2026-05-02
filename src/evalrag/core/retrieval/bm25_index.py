from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalrag.core.retrieval import Hit
from evalrag.storage.models import Chunk


class BM25Index:
    def __init__(self, session: Session) -> None:
        self.s = session

    def search(self, query: str, k: int, doc_id: str) -> list[Hit]:
        ts_query = func.plainto_tsquery("english", query)
        rank = func.ts_rank(Chunk.ts_vec, ts_query)
        stmt = (
            select(Chunk, rank.label("rk"))
            .where(Chunk.doc_id == doc_id)
            .where(Chunk.ts_vec.op("@@")(ts_query))
            .order_by(rank.desc())
            .limit(k)
        )
        rows = self.s.execute(stmt).all()
        return [Hit(chunk_id=c.chunk_id, doc_id=str(c.doc_id), text=c.text,
                    score=float(r), source="sparse")
                for c, r in rows]

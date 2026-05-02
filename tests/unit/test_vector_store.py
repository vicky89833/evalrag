import numpy as np
import pytest

from evalrag.core.retrieval import Hit
from evalrag.core.retrieval.vector_store import VectorStore
from evalrag.storage.models import Chunk, Doc

pytestmark = pytest.mark.integration


def _seed(db_session):
    doc = Doc(filename="t.txt", status="ready")
    db_session.add(doc)
    db_session.flush()
    rng = np.random.default_rng(0)
    chunks = []
    for i in range(5):
        v = rng.standard_normal(1024).astype(np.float32)
        v /= np.linalg.norm(v)
        chunks.append(Chunk(doc_id=doc.id, chunk_id=f"c{i}", text=f"text {i}",
                            embedding=v.tolist(), ts_vec="", parent_id=None, metadata_={}))
    db_session.add_all(chunks)
    db_session.commit()
    return doc, chunks


def test_search_returns_topk_in_score_order(db_session):
    doc, chunks = _seed(db_session)
    query_vec = np.array(chunks[2].embedding, dtype=np.float32)
    vs = VectorStore(db_session)
    hits = vs.search(query_vec, k=3, doc_id=str(doc.id))
    assert len(hits) == 3
    assert hits[0].chunk_id == "c2"
    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score

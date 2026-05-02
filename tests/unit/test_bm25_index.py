import pytest
from sqlalchemy import text

from evalrag.core.retrieval.bm25_index import BM25Index
from evalrag.storage.models import Chunk, Doc

pytestmark = pytest.mark.integration


def test_keyword_search_finds_match(db_session):
    doc = Doc(filename="t.txt", status="ready"); db_session.add(doc); db_session.flush()
    rows = [
        Chunk(doc_id=doc.id, chunk_id="c1", text="cats are mammals",
              embedding=[0.0]*1024, ts_vec="", parent_id=None, metadata_={}),
        Chunk(doc_id=doc.id, chunk_id="c2", text="dogs are mammals",
              embedding=[0.0]*1024, ts_vec="", parent_id=None, metadata_={}),
    ]
    db_session.add_all(rows); db_session.flush()
    db_session.execute(text("UPDATE chunks SET ts_vec = to_tsvector('english', text)"))
    db_session.commit()

    idx = BM25Index(db_session)
    hits = idx.search("cats", k=2, doc_id=str(doc.id))
    assert hits[0].chunk_id == "c1"

from unittest.mock import MagicMock

import numpy as np

from evalrag.core.retrieval import Hit
from evalrag.core.retrieval.retriever import Retriever, rrf_fuse


def test_rrf_fuse_combines_two_lists():
    dense = [Hit("a", "d", "ta", 0.9, "dense"), Hit("b", "d", "tb", 0.8, "dense")]
    sparse = [Hit("b", "d", "tb", 5.0, "sparse"), Hit("c", "d", "tc", 3.0, "sparse")]
    fused = rrf_fuse([dense, sparse], k=60, top=3)
    ids = [h.chunk_id for h in fused]
    assert ids[0] == "b"  # appears in both
    assert set(ids) == {"a", "b", "c"}


def test_retriever_calls_both_and_fuses():
    vs = MagicMock()
    bm = MagicMock()
    vs.search.return_value = [Hit("a", "d", "ta", 0.9, "dense")]
    bm.search.return_value = [Hit("b", "d", "tb", 5.0, "sparse")]
    emb = MagicMock()
    emb.embed.return_value = np.zeros((1, 1024), dtype=np.float32)
    r = Retriever(vector_store=vs, bm25=bm, embedder=emb)
    hits = r.retrieve("query", k=10, doc_id="d")
    assert len(hits) == 2
    assert {h.chunk_id for h in hits} == {"a", "b"}


def test_retriever_returns_dense_only_when_sparse_empty():
    vs = MagicMock()
    bm = MagicMock()
    vs.search.return_value = [Hit("a", "d", "ta", 0.9, "dense")]
    bm.search.return_value = []
    emb = MagicMock()
    emb.embed.return_value = np.zeros((1, 1024), dtype=np.float32)
    r = Retriever(vector_store=vs, bm25=bm, embedder=emb)
    hits = r.retrieve("q", k=10, doc_id="d")
    assert len(hits) == 1 and hits[0].chunk_id == "a"

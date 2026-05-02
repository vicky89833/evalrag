import pytest

from evalrag.core.retrieval import Hit
from evalrag.core.retrieval.reranker import Reranker


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


def test_rerank_promotes_more_relevant(reranker):
    hits = [
        Hit("a", "d", "Cats are small carnivorous mammals.", 0.5, "fused"),
        Hit("b", "d", "The Eiffel Tower is in Paris.", 0.6, "fused"),
    ]
    out = reranker.rerank("Tell me about cats.", hits, top=2)
    assert out[0].chunk_id == "a"
    assert out[0].source == "rerank"


def test_rerank_returns_topk(reranker):
    hits = [Hit(f"c{i}", "d", f"text {i}", 0.5, "fused") for i in range(5)]
    out = reranker.rerank("text 2", hits, top=3)
    assert len(out) == 3


def test_rerank_returns_empty_for_empty_input(reranker):
    assert reranker.rerank("q", [], top=5) == []

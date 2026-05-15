from unittest.mock import MagicMock

from evalrag.core.retrieval.query_transformer import QueryTransformer


def _client(*responses):
    c = MagicMock()
    c.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=r))]) for r in responses
    ]
    return c


def test_factual_returns_raw_query():
    t = QueryTransformer(client=_client("factual"))
    assert t.transform("What is X?") == ["What is X?"]


def test_multi_hop_decomposes():
    t = QueryTransformer(client=_client("multi-hop", "Q1?\nQ2?\nQ3?"))
    out = t.transform("Compare X and Y over time")
    assert out == ["Compare X and Y over time", "Q1?", "Q2?", "Q3?"]


def test_abstract_uses_hyde():
    t = QueryTransformer(client=_client("abstract", "Hypothetical answer paragraph."))
    out = t.transform("Discuss the philosophy")
    assert out == ["Discuss the philosophy", "Hypothetical answer paragraph."]


def test_unknown_classification_falls_back_to_raw():
    t = QueryTransformer(client=_client("garbage"))
    assert t.transform("X?") == ["X?"]

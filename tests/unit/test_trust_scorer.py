from unittest.mock import MagicMock

from evalrag.core.eval.trust_scorer import TrustScore, TrustScorer
from evalrag.core.generation.generator import Answer
from evalrag.core.retrieval import Hit


def _hit(i, txt="text"):
    return Hit(f"c{i}", "d", txt, 0.5, "rerank")


def _answer(text="Cats are mammals [1].", cites=(1,)):
    return Answer(text=text, citations=list(cites), tokens_in=10, tokens_out=5, cost_usd=0.001)


def test_returns_zero_citation_coverage_when_no_citations():
    j = MagicMock()
    j.judge.return_value = 1.0
    s = TrustScorer(judge=j)
    out = s.score(_answer(text="No cite.", cites=()), [_hit(1)], query="q")
    assert out.citation_coverage == 0.0
    assert out.band == "red"


def test_full_score_green_band():
    j = MagicMock()
    j.judge.return_value = 1.0
    s = TrustScorer(judge=j)
    out = s.score(_answer(text="Cats are mammals [1]."), [_hit(1, "Cats are mammals.")], query="cats?")
    assert out.faithfulness == 1.0
    assert out.overall >= 80
    assert out.band == "green"


def test_judge_failure_returns_null_score():
    j = MagicMock()
    j.judge.side_effect = RuntimeError("api down")
    s = TrustScorer(judge=j)
    out = s.score(_answer(), [_hit(1)], query="q")
    assert out is None  # caller surfaces "score unavailable"


def test_low_faithfulness_forces_red():
    j = MagicMock()
    j.judge.return_value = 0.3
    s = TrustScorer(judge=j)
    out = s.score(_answer(), [_hit(1)], query="q")
    assert out.band == "red"

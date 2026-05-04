from unittest.mock import MagicMock

from evalrag.core.generation.generator import Answer, Generator
from evalrag.core.retrieval import Hit


def _hit(i):
    return Hit(f"c{i}", "d", f"chunk text {i}", 0.5, "rerank")


def _mock_resp(text: str, tokens_in: int, tokens_out: int) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))],
        usage=MagicMock(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def test_generate_returns_answer_and_citations():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_resp(
        "Cats are mammals [1]. They purr [2].", 120, 10
    )
    g = Generator(client=client)
    out = g.generate("what are cats", [_hit(1), _hit(2)])
    assert isinstance(out, Answer)
    assert "[1]" in out.text
    assert out.citations == [1, 2]
    assert out.cost_usd > 0
    assert out.tokens_in == 120 and out.tokens_out == 10


def test_generate_extracts_only_referenced_indices():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_resp("Only one [3].", 10, 5)
    g = Generator(client=client)
    out = g.generate("q", [_hit(i) for i in range(5)])
    assert out.citations == [3]


def test_no_citations_marks_zero_coverage():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_resp("Just an answer.", 10, 5)
    g = Generator(client=client)
    out = g.generate("q", [_hit(1)])
    assert out.citations == []

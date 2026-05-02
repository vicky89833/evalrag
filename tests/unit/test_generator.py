from unittest.mock import MagicMock

from evalrag.core.generation.generator import Answer, Generator
from evalrag.core.retrieval import Hit


def _hit(i):
    return Hit(f"c{i}", "d", f"chunk text {i}", 0.5, "rerank")


def test_generate_returns_answer_and_citations():
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Cats are mammals [1]. They purr [2].")],
        usage=MagicMock(input_tokens=120, output_tokens=10),
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
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Only one [3].")],
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    g = Generator(client=client)
    out = g.generate("q", [_hit(i) for i in range(5)])
    assert out.citations == [3]


def test_no_citations_marks_zero_coverage():
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Just an answer.")],
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    g = Generator(client=client)
    out = g.generate("q", [_hit(1)])
    assert out.citations == []

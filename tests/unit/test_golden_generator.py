from unittest.mock import MagicMock

from evalrag.core.eval.golden_generator import GoldenGenerator, GoldenQA
from evalrag.core.ingest.chunker import Chunk


def _chunks(n=40):
    return [Chunk(chunk_id=f"c{i}", text=f"Chunk {i} talks about topic {i % 5}.",
                  parent_id=None, metadata={"section": f"S{i // 10}"})
            for i in range(n)]


def test_stratified_sampling_returns_n_chunks():
    g = GoldenGenerator(client=MagicMock())
    sample = g._stratified_sample(_chunks(40), n=10)
    assert len(sample) == 10
    # samples should span sections, not all from one
    sections = {c.metadata["section"] for c in sample}
    assert len(sections) > 1


def test_generate_filters_failed_validation():
    client = MagicMock()
    # First N: generation calls -> Q, then validation calls -> "yes" or "no"
    gen_resp = MagicMock(choices=[MagicMock(message=MagicMock(content="Q: What is X? A: X is Y."))])
    val_yes = MagicMock(choices=[MagicMock(message=MagicMock(content="yes"))])
    val_no = MagicMock(choices=[MagicMock(message=MagicMock(content="no"))])
    client.chat.completions.create.side_effect = [
        gen_resp, gen_resp, gen_resp,    # 3 generations
        val_yes, val_no, val_yes,        # 3 validations
    ]
    g = GoldenGenerator(client=client)
    qas = g.generate(_chunks(30), n_questions=3, n_adversarial=0)
    assert len(qas) == 2  # one dropped
    assert all(isinstance(q, GoldenQA) for q in qas)


def test_adversarial_questions_appended():
    client = MagicMock()
    gen = MagicMock(choices=[MagicMock(message=MagicMock(content="Q: What? A: A."))])
    val = MagicMock(choices=[MagicMock(message=MagicMock(content="yes"))])
    adv = MagicMock(
        choices=[MagicMock(message=MagicMock(content="What is the price of tea in Mars?"))]
    )
    client.chat.completions.create.side_effect = [gen, val, adv, adv]
    g = GoldenGenerator(client=client)
    qas = g.generate(_chunks(10), n_questions=1, n_adversarial=2)
    assert sum(1 for q in qas if q.is_adversarial) == 2

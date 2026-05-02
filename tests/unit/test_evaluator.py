from unittest.mock import patch

from evalrag.core.eval.evaluator import EvalReport, evaluate
from evalrag.core.eval.golden_generator import GoldenQA


def test_evaluate_returns_per_metric_aggregates():
    qas = [GoldenQA(question="Q1?", expected_answer="A1", source_chunk_id="c1"),
           GoldenQA(question="Q2?", expected_answer="A2", source_chunk_id="c2")]

    def fake_query(q: str) -> dict:
        return {"answer": "ans [1]", "retrieval_trace": [{"chunk_id": "c1"}],
                "trust_score": {"overall": 80,
                                "breakdown": {"faithfulness": 0.9, "context_relevance": 0.7,
                                              "citation_coverage": 1.0}},
                "cost_usd": 0.01, "latency_ms": 200}

    with patch("evalrag.core.eval.evaluator._ragas_evaluate") as m:
        m.return_value = {"faithfulness": 0.9, "answer_relevancy": 0.8,
                          "context_precision": 0.85, "context_recall": 0.7}
        rep = evaluate(qas, query_fn=fake_query)

    assert isinstance(rep, EvalReport)
    assert rep.metrics["ragas_faithfulness"] == 0.9
    assert rep.metrics["refusal_accuracy"] >= 0.0  # no adversarials → undefined→0
    assert rep.n == 2


def test_evaluate_computes_refusal_accuracy_on_adversarial():
    qas = [GoldenQA(question="Q?", expected_answer="", source_chunk_id=None,
                    is_adversarial=True)]
    REFUSE = "The document does not contain an answer to that question."

    def fake_query(q: str) -> dict:
        return {"answer": REFUSE, "retrieval_trace": [], "trust_score": None,
                "cost_usd": 0.0, "latency_ms": 50}

    with patch("evalrag.core.eval.evaluator._ragas_evaluate") as m:
        m.return_value = {"faithfulness": 0.0, "answer_relevancy": 0.0,
                          "context_precision": 0.0, "context_recall": 0.0}
        rep = evaluate(qas, query_fn=fake_query)
    assert rep.metrics["refusal_accuracy"] == 1.0

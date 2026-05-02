import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from evalrag.core.eval.golden_generator import GoldenQA
from evalrag.core.eval.regression_runner import REFUSAL_PHRASE


@dataclass
class EvalReport:
    n: int
    metrics: dict[str, float]
    per_q: list[dict] = field(default_factory=list)


def _ragas_evaluate(samples: list[dict]) -> dict[str, float]:
    """Wrap RAGAS — isolated for easy mocking."""
    from datasets import Dataset
    from ragas import evaluate as r_evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    ds = Dataset.from_list(samples)
    result = r_evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {k: float(v) for k, v in result.to_pandas().mean(numeric_only=True).items()}


def evaluate(qas: list[GoldenQA], query_fn: Callable[[str], dict]) -> EvalReport:
    samples: list[dict] = []
    per_q: list[dict] = []
    refusal_correct = 0
    n_adv = 0
    latencies: list[int] = []
    costs: list[float] = []

    for qa in qas:
        res = query_fn(qa.question)
        latencies.append(int(res.get("latency_ms", 0)))
        costs.append(float(res.get("cost_usd", 0.0)))
        if qa.is_adversarial:
            n_adv += 1
            if REFUSAL_PHRASE in res.get("answer", ""):
                refusal_correct += 1
            per_q.append({"q": qa.question, "adversarial": True,
                          "refused": REFUSAL_PHRASE in res.get("answer", "")})
            continue
        contexts = [t.get("chunk_id", "") for t in res.get("retrieval_trace", [])]
        samples.append({
            "question": qa.question,
            "answer": res.get("answer", ""),
            "contexts": [c for c in contexts if c],
            "ground_truth": qa.expected_answer,
        })
        per_q.append({"q": qa.question, "answer": res.get("answer", "")})

    ragas = _ragas_evaluate(samples) if samples else {
        "faithfulness": 0.0, "answer_relevancy": 0.0,
        "context_precision": 0.0, "context_recall": 0.0,
    }
    metrics = {
        "ragas_faithfulness": ragas["faithfulness"],
        "ragas_answer_relevance": ragas["answer_relevancy"],
        "ragas_context_precision": ragas["context_precision"],
        "ragas_context_recall": ragas["context_recall"],
        "refusal_accuracy": (refusal_correct / n_adv) if n_adv else 0.0,
        "p95_latency_ms": int(statistics.quantiles(latencies, n=20)[-1]) if len(latencies) >= 20 else max(latencies, default=0),
        "avg_cost_usd": statistics.mean(costs) if costs else 0.0,
    }
    return EvalReport(n=len(qas), metrics=metrics, per_q=per_q)

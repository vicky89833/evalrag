import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from pydantic import SecretStr

from evalrag.config import get_settings
from evalrag.core.eval.golden_generator import GoldenQA
from evalrag.core.eval.regression_runner import REFUSAL_PHRASE


@dataclass
class EvalReport:
    n: int
    metrics: dict[str, float]
    per_q: list[dict[str, Any]] = field(default_factory=list)


def _ragas_evaluate(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Wrap RAGAS — isolated for easy mocking."""
    from datasets import Dataset
    from ragas import evaluate as r_evaluate
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from evalrag.core.ingest.embedder import Embedder

    class EvalRAGEmbeddings(BaseRagasEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self._embedder = Embedder()

        def embed_query(self, text: str) -> list[float]:
            return cast(list[float], self._embedder.embed([text])[0].tolist())

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors = self._embedder.embed(texts)
            if isinstance(vectors, np.ndarray):
                return cast(list[list[float]], vectors.tolist())
            return vectors

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    s = get_settings()
    api_key = s.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("L2 evaluation requires GEMINI_API_KEY")

    from langchain_openai import ChatOpenAI

    judge_llm = ChatOpenAI(
        model=s.JUDGE_MODEL,
        api_key=SecretStr(api_key),
        base_url=s.LLM_BASE_URL,
        temperature=0.0,
        timeout=60,
        max_retries=2,
    )

    ds = Dataset.from_list(samples)
    result = r_evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=EvalRAGEmbeddings(),
        show_progress=False,
    )
    df = result.to_pandas()  # type: ignore[union-attr]
    return {k: float(v) for k, v in df.mean(numeric_only=True).items()}


def evaluate(qas: list[GoldenQA], query_fn: Callable[[str], dict[str, Any]]) -> EvalReport:
    samples: list[dict[str, Any]] = []
    per_q: list[dict[str, Any]] = []
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
        contexts = [
            t.get("text") or t.get("chunk_id", "")
            for t in res.get("retrieval_trace", [])
            if isinstance(t, dict)
        ]
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
        "p95_latency_ms": (
            int(statistics.quantiles(latencies, n=20)[-1])
            if len(latencies) >= 20
            else max(latencies, default=0)
        ),
        "avg_cost_usd": statistics.mean(costs) if costs else 0.0,
    }
    return EvalReport(n=len(qas), metrics=metrics, per_q=per_q)

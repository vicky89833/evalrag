import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REFUSAL_PHRASE = "The document does not contain an answer to that question."


@dataclass
class RegressionReport:
    n_total: int
    n_substring_pass: int
    n_refusal_pass: int
    avg_trust_overall: float
    p95_latency_ms: int
    avg_cost_usd: float
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_substring_pass_rate"] = self.n_substring_pass / self.n_total if self.n_total else 0.0
        return d

    def compare(self, baseline: dict[str, Any]) -> dict[str, Any]:
        cur = self.summary()
        return {f"{k}_delta": cur[k] - baseline[k] for k in baseline if k in cur}


class RegressionRunner:
    def __init__(self, set_path: Path, query_fn: Callable[[str], dict[str, Any]]) -> None:
        self.items = [json.loads(line) for line in Path(set_path).read_text().splitlines() if line]
        self.query_fn = query_fn

    @classmethod
    def from_items(
        cls,
        items: list[dict[str, Any]],
        query_fn: Callable[[str], dict[str, Any]],
    ) -> "RegressionRunner":
        inst = cls.__new__(cls)
        inst.items = items
        inst.query_fn = query_fn
        return inst

    def run(self) -> RegressionReport:
        per_q: list[dict[str, Any]] = []
        latencies: list[int] = []
        costs: list[float] = []
        trusts: list[float] = []
        substring_pass = refusal_pass = 0

        for item in self.items:
            t0 = time.perf_counter()
            res = self.query_fn(item["question"])
            elapsed = res.get("latency_ms") or int((time.perf_counter() - t0) * 1000)
            latencies.append(elapsed)
            costs.append(float(res.get("cost_usd", 0.0)))
            trust = res.get("trust_score") or {}
            if "overall" in trust:
                trusts.append(float(trust["overall"]))
            answered = res.get("answer", "")
            sub_ok = all(sub.lower() in answered.lower()
                         for sub in item.get("expected_answer_substrings", []))
            ref_ok = (REFUSAL_PHRASE in answered) if item.get("must_refuse") else True
            if sub_ok and not item.get("must_refuse"):
                substring_pass += 1
            if item.get("must_refuse") and ref_ok:
                refusal_pass += 1
            per_q.append({"id": item["id"], "type": item["type"], "answer": answered,
                          "substring_pass": sub_ok, "refusal_pass": ref_ok,
                          "trust": trust, "latency_ms": elapsed})

        n = len(self.items)
        return RegressionReport(
            n_total=n,
            n_substring_pass=substring_pass,
            n_refusal_pass=refusal_pass,
            avg_trust_overall=statistics.mean(trusts) if trusts else 0.0,
            p95_latency_ms=(
                int(statistics.quantiles(latencies, n=20)[-1])
                if len(latencies) >= 20
                else max(latencies, default=0)
            ),
            avg_cost_usd=statistics.mean(costs) if costs else 0.0,
            per_question=per_q,
        )

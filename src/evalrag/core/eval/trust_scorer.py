import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from evalrag.config import get_settings, make_llm_client
from evalrag.core.generation.generator import Answer
from evalrag.core.retrieval import Hit

FAITH_W, RELEV_W, CITE_W = 0.6, 0.2, 0.2


@dataclass
class TrustScore:
    overall: int
    faithfulness: float
    context_relevance: float
    citation_coverage: float
    band: str  # green | amber | red


class FaithfulnessJudge(Protocol):
    def judge(self, question: str, answer_text: str, chunks: list[str]) -> float: ...


class GPTJudge:
    client: Any

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            client = make_llm_client()
        self.client = client

    def judge(self, question: str, answer_text: str, chunks: list[str]) -> float:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        prompt = (
            "You are a strict fact-checker. Given a question, an answer, and "
            "context chunks, return ONLY a number 0.0-1.0 where 1.0 = every "
            "claim in the answer is fully supported by the chunks, and 0.0 = "
            "no claim is supported. No prose."
            f"\n\nQuestion: {question}\nAnswer: {answer_text}\n\nContext:\n{ctx}\n\nScore:"
        )
        resp = self.client.chat.completions.create(
            model=get_settings().JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=8,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"[01](?:\.\d+)?", raw)
        return float(m.group(0)) if m else 0.0


_REFUSAL_RE = re.compile(r"document does not contain|cannot answer|no answer", re.I)


def _band(
    overall: int, faithfulness: float, citation_coverage: float, answer_text: str = ""
) -> str:
    # Refusals legitimately have no citations — judge them on faithfulness alone.
    if _REFUSAL_RE.search(answer_text):
        return "green" if faithfulness >= 0.8 and overall >= 80 else "amber"
    if faithfulness < 0.5 or citation_coverage == 0.0:
        return "red"
    if overall >= 80:
        return "green"
    if overall >= 60:
        return "amber"
    return "red"


def _citation_coverage(answer_text: str, citations: list[int]) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if s]
    if not sentences:
        return 0.0
    if not citations:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return cited / len(sentences)


def _context_relevance(hits: list[Hit]) -> float:
    if not hits:
        return 0.0
    weights = np.array([1.0 / (i + 1) for i in range(len(hits))])
    scores = np.array([max(0.0, min(1.0, h.score)) for h in hits])
    return float(np.average(scores, weights=weights))


class TrustScorer:
    def __init__(self, judge: FaithfulnessJudge | None = None) -> None:
        self.judge = judge if judge is not None else GPTJudge()

    def score(self, answer: Answer, hits: list[Hit], query: str) -> TrustScore | None:
        try:
            faith = float(self.judge.judge(query, answer.text, [h.text for h in hits]))
        except Exception:
            return None
        cov = _citation_coverage(answer.text, answer.citations)
        relev = _context_relevance(hits)
        overall = int(round(100 * (FAITH_W * faith + RELEV_W * relev + CITE_W * cov)))
        return TrustScore(overall=overall, faithfulness=faith,
                          context_relevance=relev, citation_coverage=cov,
                          band=_band(overall, faith, cov, answer.text))

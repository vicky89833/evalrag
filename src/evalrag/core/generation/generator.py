import re
from dataclasses import dataclass
from typing import Any

from evalrag.config import get_settings, make_llm_client
from evalrag.core.generation.prompts import ANSWER_SYSTEM, build_user_prompt
from evalrag.core.retrieval import Hit
from evalrag.observability.tracer import trace_span

# gpt-4o pricing (as of 2026-05): $2.50/MTok in, $10/MTok out
_PRICE_IN = 2.50 / 1_000_000
_PRICE_OUT = 10.0 / 1_000_000


@dataclass
class Answer:
    text: str
    citations: list[int]
    tokens_in: int
    tokens_out: int
    cost_usd: float


class Generator:
    client: Any

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            client = make_llm_client()
        self.client = client
        self.model = get_settings().ANSWER_MODEL

    @trace_span("generate")
    def generate(self, question: str, hits: list[Hit]) -> Answer:
        user = build_user_prompt(question, [h.text for h in hits])
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        cites = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
        cost = resp.usage.prompt_tokens * _PRICE_IN + resp.usage.completion_tokens * _PRICE_OUT
        return Answer(text=text, citations=cites,
                      tokens_in=resp.usage.prompt_tokens,
                      tokens_out=resp.usage.completion_tokens,
                      cost_usd=cost)

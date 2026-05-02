import re
from dataclasses import dataclass
from typing import Protocol

from evalrag.config import get_settings
from evalrag.core.generation.prompts import ANSWER_SYSTEM, build_user_prompt
from evalrag.core.retrieval import Hit
from evalrag.observability.tracer import trace_span

# Sonnet 4.6 pricing (as of 2026-05): $3/MTok in, $15/MTok out
_PRICE_IN = 3.0 / 1_000_000
_PRICE_OUT = 15.0 / 1_000_000


@dataclass
class Answer:
    text: str
    citations: list[int]
    tokens_in: int
    tokens_out: int
    cost_usd: float


class _Anthropic(Protocol):
    def messages(self) -> object: ...


class Generator:
    def __init__(self, client: object | None = None) -> None:
        if client is None:
            from anthropic import Anthropic
            client = Anthropic()
        self.client = client
        self.model = get_settings().ANSWER_MODEL

    @trace_span("generate")
    def generate(self, question: str, hits: list[Hit]) -> Answer:
        user = build_user_prompt(question, [h.text for h in hits])
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=ANSWER_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text
        cites = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
        cost = resp.usage.input_tokens * _PRICE_IN + resp.usage.output_tokens * _PRICE_OUT
        return Answer(text=text, citations=cites,
                      tokens_in=resp.usage.input_tokens,
                      tokens_out=resp.usage.output_tokens,
                      cost_usd=cost)

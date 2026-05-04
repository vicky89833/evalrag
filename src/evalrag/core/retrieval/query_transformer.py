from typing import Any

from evalrag.config import get_settings, make_llm_client

_ROUTER = (
    "Classify the question as one of: factual, multi-hop, abstract.\n"
    "- factual: single-fact lookup\n"
    "- multi-hop: needs joining 2+ facts\n"
    "- abstract: open-ended / conceptual\n"
    "Reply with ONE word only.\n\nQuestion: {q}\nClass:"
)
_DECOMP = (
    "Break this multi-hop question into 2-3 simpler sub-questions, "
    "ONE per line, no numbering, no preamble.\n\nQuestion: {q}"
)
_HYDE = (
    "Write a short, plausible answer paragraph (2-3 sentences) for this "
    "question, even if you must guess. Output the paragraph only.\n\nQuestion: {q}"
)


class QueryTransformer:
    client: Any

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            client = make_llm_client()
        self.client = client
        self.model = get_settings().ROUTER_MODEL

    def _call(self, prompt: str, max_tokens: int = 200) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max_tokens,
        )
        text: str = r.choices[0].message.content.strip()
        return text

    def transform(self, question: str) -> list[str]:
        cls = self._call(_ROUTER.format(q=question), max_tokens=8).lower()
        if cls.startswith("multi"):
            lines = [
                ln.strip()
                for ln in self._call(_DECOMP.format(q=question)).splitlines()
                if ln.strip()
            ]
            return lines or [question]
        if cls.startswith("abstract"):
            return [self._call(_HYDE.format(q=question))]
        return [question]

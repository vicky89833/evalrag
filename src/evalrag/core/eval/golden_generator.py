import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from evalrag.config import get_settings, make_llm_client
from evalrag.core.ingest.chunker import Chunk

_GEN_PROMPT = (
    "Read the chunk and write ONE factual question whose answer is found "
    "ONLY in this chunk. Do not use phrases like 'according to the text'. "
    "Output exactly:\nQ: <question>\nA: <one-sentence answer>\n\nCHUNK:\n{text}"
)
_VAL_PROMPT = (
    "Question: {q}\nProposed answer: {a}\nChunk: {text}\n\n"
    "Is the answer fully supported by the chunk AND does the question avoid "
    "phrases like 'according to the text'? Reply ONLY 'yes' or 'no'."
)
_ADV_PROMPT = (
    "Write ONE question that is plausibly related to a document about "
    "{topic_hint} but whose answer is NOT in the document. Output the "
    "question only, no preamble."
)


@dataclass
class GoldenQA:
    question: str
    expected_answer: str
    source_chunk_id: str | None
    is_adversarial: bool = False


class GoldenGenerator:
    client: Any

    def __init__(self, client: object | None = None, model: str | None = None) -> None:
        if client is None:
            client = make_llm_client()
        self.client = client
        self.model = model or get_settings().GOLDEN_MODEL

    def _stratified_sample(self, chunks: list[Chunk], n: int) -> list[Chunk]:
        if len(chunks) <= n:
            return list(chunks)
        groups: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            groups[c.metadata.get("section", "_")].append(c)
        out: list[Chunk] = []
        per_group = max(1, n // max(1, len(groups)))
        for grp in groups.values():
            step = max(1, len(grp) // per_group)
            out.extend(grp[::step][:per_group])
        # if undershoot, fill from any remaining chunks not yet selected
        if len(out) < n:
            seen = {id(c) for c in out}
            for c in chunks:
                if id(c) not in seen:
                    out.append(c)
                    if len(out) >= n:
                        break
        return out[:n]

    def _call(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=200,
        )
        text: str = r.choices[0].message.content.strip()
        return text

    def _parse_qa(self, text: str) -> tuple[str, str] | None:
        m = re.search(r"Q:\s*(.+?)\s*A:\s*(.+)", text, re.S)
        if not m:
            return None
        return m.group(1).strip(), m.group(2).strip()

    def generate(self, chunks: list[Chunk], n_questions: int = 30,
                 n_adversarial: int = 5) -> list[GoldenQA]:
        sample = self._stratified_sample(chunks, n=n_questions)
        candidates: list[tuple[GoldenQA, str]] = []
        for c in sample:
            raw = self._call(_GEN_PROMPT.format(text=c.text))
            parsed = self._parse_qa(raw)
            if parsed is None:
                continue
            candidates.append((GoldenQA(question=parsed[0], expected_answer=parsed[1],
                                        source_chunk_id=c.chunk_id), c.text))

        survivors: list[GoldenQA] = []
        for cand, ctext in candidates:
            verdict = self._call(
                _VAL_PROMPT.format(q=cand.question, a=cand.expected_answer, text=ctext)
            )
            if verdict.lower().startswith("yes"):
                survivors.append(cand)

        topic_hint = (chunks[0].text[:80] if chunks else "the document").replace("\n", " ")
        for _ in range(n_adversarial):
            q = self._call(_ADV_PROMPT.format(topic_hint=topic_hint))
            survivors.append(GoldenQA(question=q, expected_answer="",
                                      source_chunk_id=None, is_adversarial=True))
        return survivors

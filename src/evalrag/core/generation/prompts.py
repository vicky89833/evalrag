ANSWER_SYSTEM = """You answer questions about a single document.

Rules:
1. Use ONLY the numbered context chunks. Do not use outside knowledge.
2. Cite every sentence with one or more inline markers like [1], [2].
3. If the chunks do not contain the answer, reply exactly: "The document does not contain an answer to that question."
4. Never apologize or hedge. Be direct.
"""


def build_user_prompt(question: str, chunks: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return f"{ctx}\n\n---\n\nQuestion: {question}\n\nAnswer (with [n] citations):"

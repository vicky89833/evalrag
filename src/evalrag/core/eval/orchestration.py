import logging
from typing import Any
from uuid import UUID

import httpx

from evalrag.core.eval.evaluator import evaluate
from evalrag.core.eval.golden_generator import GoldenGenerator
from evalrag.core.ingest.chunker import Chunk as DomainChunk
from evalrag.storage.db import SessionLocal
from evalrag.storage.models import Chunk, Doc, EvalRun, Golden

log = logging.getLogger(__name__)


def run_l2(doc_id: UUID, api_base: str = "http://localhost:8000",
           git_sha: str = "unknown") -> None:
    with SessionLocal() as s:
        doc = s.get(Doc, doc_id)
        if doc is None:
            return
        rows = s.query(Chunk).filter_by(doc_id=doc_id).all()
        chunks = [DomainChunk(chunk_id=r.chunk_id, text=r.text,
                              parent_id=r.parent_id, metadata=r.metadata_)
                  for r in rows]

    try:
        gen = GoldenGenerator()
        qas = gen.generate(chunks, n_questions=30, n_adversarial=5)
    except Exception as e:
        log.exception("golden gen failed: %s", e)
        with SessionLocal() as s:
            d = s.get(Doc, doc_id)
            if d is not None:
                d.status = "eval_failed"
                s.commit()
        return

    with SessionLocal() as s:
        for qa in qas:
            s.add(Golden(doc_id=doc_id, question=qa.question,
                         expected_answer_chunks=[qa.source_chunk_id] if qa.source_chunk_id else [],
                         is_adversarial=qa.is_adversarial))
        s.commit()
        chunk_text_by_id = {
            row.chunk_id: row.text
            for row in s.query(Chunk.chunk_id, Chunk.text).filter_by(doc_id=doc_id).all()
        }

    def query_fn(q: str) -> dict[str, Any]:
        r = httpx.post(f"{api_base}/query",
                       json={"doc_id": str(doc_id), "question": q}, timeout=60)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        for hit in data.get("retrieval_trace", []):
            if isinstance(hit, dict) and "chunk_id" in hit:
                hit["text"] = chunk_text_by_id.get(hit["chunk_id"], "")
        return data

    try:
        report = evaluate(qas, query_fn=query_fn)
    except Exception as e:
        log.exception("L2 evaluate failed: %s", e)
        return

    with SessionLocal() as s:
        s.add(EvalRun(doc_id=doc_id, layer="L2", metrics=report.metrics,
                      git_sha=git_sha, config={}))
        d = s.get(Doc, doc_id)
        if d is not None:
            d.eval_summary = report.metrics
            d.status = "ready"
        s.commit()

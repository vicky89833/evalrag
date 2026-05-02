import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from evalrag.api.deps import (get_embedder, get_generator,
                              get_query_transformer, get_reranker,
                              get_session_dep, get_trust_scorer)
from evalrag.config import get_settings
from evalrag.core.eval.trust_scorer import TrustScorer
from evalrag.core.generation.generator import Generator
from evalrag.core.ingest.embedder import Embedder
from evalrag.core.retrieval import Hit
from evalrag.core.retrieval.bm25_index import BM25Index
from evalrag.core.retrieval.query_transformer import QueryTransformer
from evalrag.core.retrieval.reranker import Reranker
from evalrag.core.retrieval.retriever import Retriever
from evalrag.core.retrieval.vector_store import VectorStore
from evalrag.storage.models import Doc, QueryLog

router = APIRouter()


class QueryReq(BaseModel):
    doc_id: UUID
    question: str
    use_reranker: bool = True


@router.post("/query")
def query(
    req: QueryReq,
    session: Session = Depends(get_session_dep),
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
    generator: Generator = Depends(get_generator),
    trust_scorer: TrustScorer = Depends(get_trust_scorer),
    transformer: QueryTransformer = Depends(get_query_transformer),
) -> dict:
    s = get_settings()
    if session.get(Doc, req.doc_id) is None:
        raise HTTPException(404, "doc not found")

    started = time.perf_counter()
    retriever = Retriever(VectorStore(session), BM25Index(session), embedder)
    variants = transformer.transform(req.question)
    all_hits: dict[str, Hit] = {}
    for v in variants:
        for h in retriever.retrieve(v, k=s.TOP_K_RETRIEVE, doc_id=str(req.doc_id)):
            all_hits.setdefault(h.chunk_id, h)
    fused = list(all_hits.values())[:s.TOP_K_RETRIEVE]

    if not fused:
        return {
            "answer": "The document does not contain an answer to that question.",
            "citations": [], "retrieval_trace": [], "latency_ms": 0,
            "cost_usd": 0.0, "trust_score": None,
        }

    top = (reranker.rerank(req.question, fused, top=s.TOP_K_RERANK)
           if req.use_reranker else fused[:s.TOP_K_RERANK])
    answer = generator.generate(req.question, top)
    elapsed = int((time.perf_counter() - started) * 1000)

    trust = trust_scorer.score(answer, top, req.question)
    trust_payload = (
        {"overall": trust.overall, "band": trust.band,
         "breakdown": {"faithfulness": trust.faithfulness,
                       "context_relevance": trust.context_relevance,
                       "citation_coverage": trust.citation_coverage}}
        if trust else None
    )

    trace = [{"chunk_id": h.chunk_id, "score": h.score, "source": h.source} for h in top]
    log = QueryLog(doc_id=req.doc_id, question=req.question, answer=answer.text,
                   trust_score=trust_payload, retrieval_trace={"top": trace},
                   latency_ms=elapsed, cost_usd=answer.cost_usd)
    session.add(log)
    session.commit()

    return {
        "answer": answer.text,
        "citations": answer.citations,
        "retrieval_trace": trace,
        "latency_ms": elapsed,
        "cost_usd": answer.cost_usd,
        "trust_score": trust_payload,
    }

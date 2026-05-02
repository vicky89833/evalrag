from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (JSON, TIMESTAMP, ForeignKey, Index, Integer, String,
                        Text, func)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from evalrag.config import get_settings

EMBED_DIM = get_settings().EMBED_DIM


class Base(DeclarativeBase):
    pass


class Doc(Base):
    __tablename__ = "docs"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    uploaded_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(String(32), default="pending", server_default="pending")
    eval_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    doc_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("docs.id", ondelete="CASCADE"), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    ts_vec: Mapped[str] = mapped_column(TSVECTOR)
    parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, server_default="{}")


Index("ix_chunks_embedding_hnsw", Chunk.embedding,
      postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64},
      postgresql_ops={"embedding": "vector_cosine_ops"})
Index("ix_chunks_ts_vec", Chunk.ts_vec, postgresql_using="gin")


class Golden(Base):
    __tablename__ = "goldens"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("docs.id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text)
    expected_answer_chunks: Mapped[list[str]] = mapped_column(JSON)
    is_adversarial: Mapped[bool] = mapped_column(default=False, server_default=func.false())


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), ForeignKey("docs.id", ondelete="CASCADE"), nullable=True)
    run_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    layer: Mapped[str] = mapped_column(String(8))
    metrics: Mapped[dict] = mapped_column(JSON)
    git_sha: Mapped[str] = mapped_column(String(40))
    config: Mapped[dict] = mapped_column(JSON)


class QueryLog(Base):
    __tablename__ = "query_log"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("docs.id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    trust_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieval_trace: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column()
    ts: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

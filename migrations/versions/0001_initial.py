"""initial schema with pgvector + tsvector"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBED_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table("docs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("uploaded_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("eval_summary", sa.JSON, nullable=True),
    )

    op.create_table("chunks",
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("docs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("chunk_id", sa.String(64), primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        sa.Column("ts_vec", TSVECTOR, nullable=False),
        sa.Column("parent_id", sa.String(64), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=False, server_default="{}"),
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX ix_chunks_ts_vec ON chunks USING gin (ts_vec)")

    op.create_table("goldens",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("expected_answer_chunks", sa.JSON, nullable=False),
        sa.Column("is_adversarial", sa.Boolean, nullable=False, server_default=sa.false()),
    )

    op.create_table("eval_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("run_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("layer", sa.String(8), nullable=False),
        sa.Column("metrics", sa.JSON, nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=False),
        sa.Column("config", sa.JSON, nullable=False),
    )

    op.create_table("query_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("doc_id", UUID(as_uuid=True), sa.ForeignKey("docs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("trust_score", sa.JSON, nullable=True),
        sa.Column("retrieval_trace", sa.JSON, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Float, nullable=False),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("query_log")
    op.drop_table("eval_runs")
    op.drop_table("goldens")
    op.drop_table("chunks")
    op.drop_table("docs")

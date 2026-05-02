# EvalRAG

Single-doc RAG with live trust scores and auto-generated eval sets. See `docs/superpowers/specs/2026-05-01-evalrag-design.md`.

## Quick start
1. `cp .env.example .env` and fill in API keys
2. `make up` — start Postgres + Langfuse
3. `make install`
4. `make migrate`
5. `make api` (terminal 1) and `make ui` (terminal 2)

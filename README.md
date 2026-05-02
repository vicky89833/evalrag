# EvalRAG

Single-doc RAG with live trust scores and auto-generated eval sets. See `docs/superpowers/specs/2026-05-01-evalrag-design.md`.

## Prerequisites
- Python 3.12
- PostgreSQL 16 with [pgvector](https://github.com/pgvector/pgvector) installed locally
- A user `evalrag` and database `evalrag` (see "Postgres setup" below)

## Quick start
1. `python3.12 -m venv .venv` (one-time)
2. `cp .env.example .env` — fill in `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`; adjust `DATABASE_URL` if your Postgres differs
3. `make install`
4. `make migrate`
5. `make api` (terminal 1) and `make ui` (terminal 2)

## Postgres setup (one-time)
```bash
psql -h localhost -U postgres -d postgres -c "CREATE USER evalrag WITH PASSWORD 'evalrag' SUPERUSER;"
psql -h localhost -U postgres -d postgres -c "CREATE DATABASE evalrag OWNER evalrag;"
psql -h localhost -U postgres -d evalrag    -c "CREATE EXTENSION vector;"
```
If pgvector isn't packaged for your Postgres install, build from source against your `pg_config` — see https://github.com/pgvector/pgvector#installation.

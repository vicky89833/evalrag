.PHONY: install lint test test-int up down migrate ui api regression

install:
	. .venv/bin/activate && pip install -e ".[dev]"

lint:
	. .venv/bin/activate && ruff check src tests && mypy src

test:
	. .venv/bin/activate && pytest -m "not integration and not llm"

test-int:
	. .venv/bin/activate && pytest -m integration

up:
	docker compose up -d

down:
	docker compose down

migrate:
	. .venv/bin/activate && alembic upgrade head

api:
	. .venv/bin/activate && uvicorn evalrag.api.main:app --reload --port 8000

ui:
	. .venv/bin/activate && streamlit run src/evalrag/ui/streamlit_app.py

regression:
	. .venv/bin/activate && python scripts/run_regression.py

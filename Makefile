PYTHON ?= python3.12
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
INSTALLED := $(VENV)/.installed

.PHONY: venv install lint test test-int migrate ui api regression clean-venv

venv: $(VENV_PY)

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)

$(INSTALLED): pyproject.toml $(VENV_PY)
	$(VENV_PIP) install -e ".[dev]"
	touch $(INSTALLED)

install: $(INSTALLED)

lint: $(INSTALLED)
	$(VENV_PY) -m ruff check src tests
	$(VENV_PY) -m mypy src

test: $(INSTALLED)
	$(VENV_PY) -m pytest -m "not integration and not llm"

test-int: $(INSTALLED)
	$(VENV_PY) -m pytest -m integration

migrate: $(INSTALLED)
	$(VENV_PY) -m alembic upgrade head

api: $(INSTALLED)
	$(VENV_PY) -m uvicorn evalrag.api.main:app --reload --port 8000

ui: $(INSTALLED)
	$(VENV_PY) -m streamlit run src/evalrag/ui/streamlit_app.py

regression: $(INSTALLED)
	$(VENV_PY) scripts/run_regression.py

clean-venv:
	rm -rf $(VENV)

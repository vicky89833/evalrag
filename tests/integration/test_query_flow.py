from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from evalrag.api.deps import get_generator, get_session_dep, get_trust_scorer
from evalrag.api.main import app
from evalrag.core.eval.trust_scorer import TrustScore
from evalrag.core.generation.generator import Answer

pytestmark = pytest.mark.integration

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_session_dep] = lambda: db_session

    fake = MagicMock()
    fake.generate.return_value = Answer(
        text="Cats are mammals [1].", citations=[1],
        tokens_in=100, tokens_out=10, cost_usd=0.001,
    )
    app.dependency_overrides[get_generator] = lambda: fake

    fake_trust = MagicMock()
    fake_trust.score.return_value = TrustScore(
        overall=88, faithfulness=0.95, context_relevance=0.8,
        citation_coverage=1.0, band="green",
    )
    app.dependency_overrides[get_trust_scorer] = lambda: fake_trust
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_returns_answer_and_trace(client):
    files = {"file": ("short.txt", FIX.joinpath("short.txt").read_bytes(), "text/plain")}
    doc = client.post("/docs", files=files).json()
    r = client.post("/query", json={"doc_id": doc["id"], "question": "what about cats?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Cats")
    assert body["citations"] == [1]
    assert "retrieval_trace" in body
    assert body["latency_ms"] >= 0
    assert body["trust_score"]["overall"] == 88
    assert body["trust_score"]["band"] == "green"

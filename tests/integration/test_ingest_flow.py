from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from evalrag.api.main import app, get_session_dep
from evalrag.storage.models import Chunk, Doc

pytestmark = pytest.mark.integration

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_session_dep] = lambda: db_session
    with patch("evalrag.api.routes.docs.run_l2"):
        yield TestClient(app)
    app.dependency_overrides.clear()


def test_upload_txt_creates_doc_and_chunks(client, db_session):
    files = {"file": ("short.txt", FIX.joinpath("short.txt").read_bytes(), "text/plain")}
    r = client.post("/docs", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ingested"
    doc_id = body["id"]
    chunks = db_session.query(Chunk).filter_by(doc_id=doc_id).all()
    assert len(chunks) >= 1
    assert all(c.embedding is not None for c in chunks)


def test_upload_unsupported_returns_415(client):
    r = client.post("/docs", files={"file": ("x.xyz", b"nope", "application/octet-stream")})
    assert r.status_code == 415


def test_upload_too_large_returns_413(client, monkeypatch):
    from evalrag.config import get_settings
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_MB", 0)
    r = client.post("/docs", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 413

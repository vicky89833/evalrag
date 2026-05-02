from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from evalrag.api.deps import get_session_dep
from evalrag.api.main import app

pytestmark = pytest.mark.integration
FIX = Path(__file__).parent.parent / "fixtures"


def test_upload_schedules_l2(db_session):
    app.dependency_overrides[get_session_dep] = lambda: db_session
    client = TestClient(app)
    with patch("evalrag.api.routes.docs.run_l2") as m:
        files = {"file": ("short.txt", FIX.joinpath("short.txt").read_bytes(), "text/plain")}
        r = client.post("/docs", files=files)
        assert r.status_code == 200
        assert r.json()["eval_status"] == "pending"
        m.assert_called_once()
    app.dependency_overrides.clear()

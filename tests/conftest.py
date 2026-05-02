import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from evalrag.config import get_settings
from evalrag.storage.models import Base

os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")


@pytest.fixture(scope="session")
def admin_engine():
    base = get_settings().DATABASE_URL.rsplit("/", 1)[0]
    return create_engine(f"{base}/postgres", isolation_level="AUTOCOMMIT")


@pytest.fixture
def db_session(admin_engine):
    test_db = f"evalrag_test_{uuid.uuid4().hex[:8]}"
    with admin_engine.connect() as c:
        c.execute(text(f'CREATE DATABASE "{test_db}"'))
    base = get_settings().DATABASE_URL.rsplit("/", 1)[0]
    url = f"{base}/{test_db}"
    eng = create_engine(url)
    with eng.connect() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        c.commit()
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, expire_on_commit=False)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        eng.dispose()
        with admin_engine.connect() as c:
            c.execute(text(f'DROP DATABASE "{test_db}"'))

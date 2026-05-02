from collections.abc import Iterator

from sqlalchemy.orm import Session

from evalrag.core.ingest.embedder import Embedder
from evalrag.storage.db import SessionLocal

_embedder: Embedder | None = None


def get_session_dep() -> Iterator[Session]:
    with SessionLocal() as s:
        yield s


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

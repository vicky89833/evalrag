import threading
from collections.abc import Iterator

from sqlalchemy.orm import Session

from evalrag.core.generation.generator import Generator
from evalrag.core.ingest.embedder import Embedder
from evalrag.core.retrieval.reranker import Reranker
from evalrag.storage.db import SessionLocal

_embedder: Embedder | None = None
_embedder_lock = threading.Lock()
_reranker: Reranker | None = None
_reranker_lock = threading.Lock()
_generator: Generator | None = None
_generator_lock = threading.Lock()


def get_session_dep() -> Iterator[Session]:
    with SessionLocal() as s:
        yield s


def get_embedder() -> Embedder:
    """Lazy singleton with double-checked locking — FastAPI runs sync deps
    in a threadpool, so concurrent first-requests would otherwise build the
    BGE model twice (~5s + ~1.3GB VRAM each)."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                _generator = Generator()
    return _generator

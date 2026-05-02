from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from evalrag.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_settings().DATABASE_URL, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
    )


def SessionLocal() -> Session:  # noqa: N802 — preserves call-site `SessionLocal()` ergonomics
    return get_sessionmaker()()


def get_session() -> Iterator[Session]:
    with SessionLocal() as s:
        yield s

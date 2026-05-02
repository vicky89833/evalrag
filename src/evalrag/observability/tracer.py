import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import structlog

P = ParamSpec("P")
R = TypeVar("R")

_log = structlog.get_logger("evalrag.span")


def trace_span(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                _log.info(
                    "span",
                    name=name,
                    ms=int((time.perf_counter() - t0) * 1000),
                    status="ok",
                )
                return result
            except Exception as e:
                _log.warning(
                    "span",
                    name=name,
                    ms=int((time.perf_counter() - t0) * 1000),
                    status="error",
                    error=str(e),
                )
                raise

        return wrapper

    return deco

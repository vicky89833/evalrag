from fastapi import FastAPI

from evalrag.api.deps import get_session_dep
from evalrag.api.routes.docs import router as docs_router
from evalrag.api.routes.query import router as query_router

app = FastAPI(title="EvalRAG", version="0.1.0")
app.include_router(docs_router)
app.include_router(query_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["app", "get_session_dep"]

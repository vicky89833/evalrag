import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from evalrag.api.deps import get_embedder, get_session_dep
from evalrag.config import get_settings
from evalrag.core.eval.orchestration import run_l2
from evalrag.core.ingest.chunker import ChunkerError
from evalrag.core.ingest.chunker import chunk as chunk_doc
from evalrag.core.ingest.embedder import Embedder
from evalrag.core.ingest.loader import LoaderError, load
from evalrag.storage.models import Chunk, Doc

router = APIRouter()

_FILE_DEP = File(...)
_SESSION_DEP = Depends(get_session_dep)
_EMBEDDER_DEP = Depends(get_embedder)


@router.post("/docs")
async def upload(
    background: BackgroundTasks,
    file: UploadFile = _FILE_DEP,
    session: Session = _SESSION_DEP,
    embedder: Embedder = _EMBEDDER_DEP,
) -> dict[str, object]:
    s = get_settings()
    raw = await file.read()
    if len(raw) > s.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file exceeds {s.MAX_UPLOAD_MB} MB")

    suffix = Path(file.filename or "").suffix.lower() or ".txt"
    # NOTE: tempfile cleanup deferred — see plan §6 / Task 7.x. Files in /tmp
    # are flushed by macOS on reboot; acceptable for the demo's lifetime.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        document = load(tmp_path)
    except LoaderError as e:
        msg = str(e)
        if "unsupported" in msg:
            raise HTTPException(415, msg) from e
        raise HTTPException(400, msg) from e

    try:
        chunks = chunk_doc(document)
    except ChunkerError as e:
        raise HTTPException(400, str(e)) from e

    vectors = embedder.embed([c.text for c in chunks])

    safe_name = Path(file.filename or tmp_path.name).name  # basename only
    doc = Doc(filename=safe_name, status="ready")
    session.add(doc)
    session.flush()

    for c, v in zip(chunks, vectors, strict=True):
        session.add(Chunk(doc_id=doc.id, chunk_id=c.chunk_id, text=c.text,
                          embedding=v.tolist(), parent_id=c.parent_id,
                          metadata_=c.metadata, ts_vec=""))
    session.commit()
    background.add_task(run_l2, doc.id)
    return {"id": str(doc.id), "filename": doc.filename, "chunks": len(chunks),
            "status": "ingested", "eval_status": "pending"}


@router.get("/docs/{doc_id}")
def get_doc(doc_id: UUID, session: Session = _SESSION_DEP) -> dict[str, object]:
    d = session.get(Doc, doc_id)
    if d is None:
        raise HTTPException(404, "not found")
    return {"id": str(d.id), "filename": d.filename, "status": d.status,
            "eval_summary": d.eval_summary}

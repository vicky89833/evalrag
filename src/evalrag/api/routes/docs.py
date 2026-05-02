import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from evalrag.api.deps import get_embedder, get_session_dep
from evalrag.config import get_settings
from evalrag.core.ingest.chunker import ChunkerError, chunk as chunk_doc
from evalrag.core.ingest.embedder import Embedder
from evalrag.core.ingest.loader import LoaderError, load
from evalrag.storage.models import Chunk, Doc

router = APIRouter()


@router.post("/docs")
async def upload(
    file: UploadFile = File(...),
    session: Session = Depends(get_session_dep),
    embedder: Embedder = Depends(get_embedder),
) -> dict:
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
    return {"id": str(doc.id), "filename": doc.filename, "chunks": len(chunks), "status": "ready"}


@router.get("/docs/{doc_id}")
def get_doc(doc_id: UUID, session: Session = Depends(get_session_dep)) -> dict:
    d = session.get(Doc, doc_id)
    if d is None:
        raise HTTPException(404, "not found")
    return {"id": str(d.id), "filename": d.filename, "status": d.status,
            "eval_summary": d.eval_summary}

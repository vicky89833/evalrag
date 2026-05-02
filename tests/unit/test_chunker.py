from evalrag.core.ingest.chunker import Chunk, chunk
from evalrag.core.ingest.loader import Document


def test_chunks_long_text_into_multiple():
    text = " ".join(["sentence"] * 2000)
    chunks = chunk(Document(text=text, metadata={"filename": "x.txt", "ext": ".txt"}))
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.text for c in chunks)
    assert all(c.chunk_id for c in chunks)


def test_short_text_is_one_chunk():
    chunks = chunk(Document(text="Just a sentence.", metadata={"filename": "x.txt", "ext": ".txt"}))
    assert len(chunks) == 1


def test_md_uses_section_aware_chunker_when_headings_present():
    md = "# A\n\n" + ("alpha " * 200) + "\n\n# B\n\n" + ("beta " * 200)
    chunks = chunk(Document(text=md, metadata={"filename": "x.md", "ext": ".md"}))
    assert len(chunks) >= 2
    # at least one chunk per heading
    text_blob = " ".join(c.text for c in chunks)
    assert "alpha" in text_blob and "beta" in text_blob


def test_chunker_raises_on_empty():
    import pytest
    from evalrag.core.ingest.chunker import ChunkerError
    with pytest.raises(ChunkerError):
        chunk(Document(text="", metadata={"filename": "x", "ext": ".txt"}))

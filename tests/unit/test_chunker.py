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
    # Section boundaries respected: no chunk should contain BOTH alpha and beta
    assert not any("alpha" in c.text and "beta" in c.text for c in chunks)
    # MD branch wires parent_id to the heading node
    assert all(c.parent_id is not None for c in chunks)
    # `section` metadata is populated for stratified sampling (Task 5.1)
    sections = {c.metadata.get("section") for c in chunks}
    assert "A" in sections and "B" in sections


def test_plain_chunks_have_default_section():
    chunks = chunk(Document(text=" ".join(["w"] * 2000),
                            metadata={"filename": "x.txt", "ext": ".txt"}))
    assert all(c.metadata.get("section") == "_default" for c in chunks)
    assert all(c.parent_id is None for c in chunks)


def test_chunker_raises_on_empty():
    import pytest
    from evalrag.core.ingest.chunker import ChunkerError
    with pytest.raises(ChunkerError):
        chunk(Document(text="", metadata={"filename": "x", "ext": ".txt"}))

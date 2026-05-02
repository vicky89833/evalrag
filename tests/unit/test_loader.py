from pathlib import Path

import pytest

from evalrag.core.ingest.loader import LoaderError, load

FIX = Path(__file__).parent.parent / "fixtures"


def test_loads_txt():
    doc = load(FIX / "short.txt")
    assert "Hello world" in doc.text
    assert doc.metadata["filename"] == "short.txt"


def test_loads_md():
    doc = load(FIX / "short.md")
    assert "Title" in doc.text


def test_loads_pdf():
    doc = load(FIX / "short.pdf")
    assert "Hello world" in doc.text


def test_rejects_unsupported_type(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("nope")
    with pytest.raises(LoaderError, match="unsupported"):
        load(p)


def test_rejects_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(LoaderError, match="empty"):
        load(p)

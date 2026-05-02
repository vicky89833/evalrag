import numpy as np
import pytest

from evalrag.core.ingest.embedder import Embedder


@pytest.fixture(scope="module")
def emb():
    return Embedder()


def test_embed_returns_correct_shape(emb):
    out = emb.embed(["hello", "world"])
    assert out.shape == (2, 1024)


def test_embed_is_normalized(emb):
    out = emb.embed(["hello"])
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-3


def test_embed_handles_empty_list(emb):
    out = emb.embed([])
    assert out.shape == (0, 1024)

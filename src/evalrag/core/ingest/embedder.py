from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from evalrag.config import get_settings


class Embedder:
    def __init__(self) -> None:
        s = get_settings()
        self._model = SentenceTransformer(s.EMBED_MODEL)
        self._dim = s.EMBED_DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._model.encode(  # type: ignore[no-any-return]
            list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=32
        )

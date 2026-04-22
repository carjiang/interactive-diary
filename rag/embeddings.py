from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def embed_text(text: str, client: Optional[OpenAI] = None) -> np.ndarray:
    from openai import OpenAI as _OpenAI
    client = client or _OpenAI()
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector

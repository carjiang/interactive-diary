from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import faiss

from rag.store import append_record, load_all_records
from rag.embeddings import embed_text, EMBED_DIM

if TYPE_CHECKING:
    from openai import OpenAI

# Patchable in tests via patch.object(retriever_mod, "INDEX_DIR", ...)
INDEX_DIR = os.path.dirname(__file__)

_HYPOTHESIS_SYSTEM = (
    "You are a diary therapist. In one concise sentence, describe the writer's "
    "core emotional or mental state based on this diary entry. Output only the sentence."
)


def _index_path(user_id: str) -> str:
    return os.path.join(INDEX_DIR, f"diary_{user_id}.faiss")


def _load_index(user_id: str) -> faiss.IndexFlatIP:
    path = _index_path(user_id)
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexFlatIP(EMBED_DIM)


def _save_index(index: faiss.IndexFlatIP, user_id: str) -> None:
    faiss.write_index(index, _index_path(user_id))


def _generate_hypothesis(raw_text: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _HYPOTHESIS_SYSTEM},
            {"role": "user", "content": raw_text},
        ],
    )
    return response.choices[0].message.content.strip()


class HypothesisRetriever:
    def __init__(self, client: Optional[OpenAI] = None):
        if client is None:
            from openai import OpenAI as _OpenAI
            client = _OpenAI()
        self.client = client

    def add_entry(
        self,
        entry_id: str,
        timestamp: datetime,
        raw_text: str,
        user_id: str,
    ) -> None:

        append_record(entry_id, timestamp, raw_text, user_id)
        vector = embed_text(raw_text, self.client)
        index = _load_index(user_id)
        index.add(vector.reshape(1, -1))
        _save_index(index, user_id)

    def retrieve_similar(
        self,
        query_text: str,
        user_id: str,
        top_k: int = 3,
    ) -> list[dict]:
        records = load_all_records(user_id=user_id)
        index = _load_index(user_id)

        if not records or index.ntotal == 0:
            return []

        if len(records) != index.ntotal:
            n = min(len(records), index.ntotal)
            warnings.warn(
                f"diary_store.jsonl has {len(records)} records for user {user_id!r} "
                f"but FAISS index has {index.ntotal} vectors — truncating to {n}."
            )
            records = records[:n]

        query_vector = embed_text(query_text, self.client)
        top_k = min(top_k, len(records))
        scores, indices = index.search(query_vector.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = dict(records[idx])
            record["similarity_score"] = float(score)
            results.append(record)

        return results

from __future__ import annotations

import sys
import warnings
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from rag.embeddings import EMBED_DIM
import rag.retriever as retriever_mod
import rag.store as store_mod
from rag.retriever import HypothesisRetriever, _load_index, _save_index, _index_path


USER_A = "user-alice"
USER_B = "user-bob"


def _unit_vector(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_client_cycle(vectors: list[np.ndarray], hypothesis: str = "Feeling reflective.") -> MagicMock:
    """Client whose embeddings.create cycles through vectors; hypothesis LLM call returns fixed string."""
    embed_idx = [0]

    def embed_side_effect(input, model):
        embedding_obj = MagicMock()
        embedding_obj.embedding = vectors[embed_idx[0] % len(vectors)].tolist()
        response = MagicMock()
        response.data = [embedding_obj]
        embed_idx[0] += 1
        return response

    completion_response = MagicMock()
    completion_response.choices[0].message.content = hypothesis

    client = MagicMock()
    client.embeddings.create.side_effect = embed_side_effect
    client.chat.completions.create.return_value = completion_response
    return client


def _make_single_vector_client(vector: np.ndarray, hypothesis: str = "Feeling reflective.") -> MagicMock:
    return _make_client_cycle([vector], hypothesis)


def _mock_openai_module():
    mock_mod = MagicMock()
    mock_mod.OpenAI.return_value = MagicMock()
    return mock_mod


@pytest.fixture()
def isolated_rag(tmp_path):
    store_path = str(tmp_path / "diary_store.jsonl")
    mock_openai = _mock_openai_module()
    with ExitStack() as stack:
        stack.enter_context(patch.object(retriever_mod, "INDEX_DIR", str(tmp_path)))
        stack.enter_context(patch.object(store_mod, "STORE_PATH", store_path))
        stack.enter_context(patch.dict(sys.modules, {"openai": mock_openai}))
        yield str(tmp_path), store_path


def test_load_index_creates_fresh_index_when_missing(isolated_rag):
    index = _load_index(USER_A)
    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 0


def test_per_user_index_files_are_separate(isolated_rag):
    tmp_dir, _ = isolated_rag
    index_a = faiss.IndexFlatIP(EMBED_DIM)
    index_b = faiss.IndexFlatIP(EMBED_DIM)
    index_a.add(_unit_vector(1).reshape(1, -1))
    _save_index(index_a, USER_A)
    _save_index(index_b, USER_B)

    loaded_a = _load_index(USER_A)
    loaded_b = _load_index(USER_B)
    assert loaded_a.ntotal == 1
    assert loaded_b.ntotal == 0


def test_save_and_load_index_roundtrip(isolated_rag, tmp_path):
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(_unit_vector(42).reshape(1, -1))
    _save_index(index, USER_A)
    loaded = _load_index(USER_A)
    assert loaded.ntotal == 1


def test_add_entry_stores_record_and_vector(isolated_rag):
    client = _make_single_vector_client(_unit_vector(1), "Writer feels motivated.")
    retriever = HypothesisRetriever(client=client)

    retriever.add_entry("entry-1", datetime(2026, 4, 20, 10, 0), "Had a productive morning.", USER_A)

    records = store_mod.load_all_records(user_id=USER_A)
    assert len(records) == 1
    assert records[0]["entry_id"] == "entry-1"
    assert records[0]["user_id"] == USER_A
    assert records[0]["hypothesis"] == "Writer feels motivated."
    assert _load_index(USER_A).ntotal == 1


def test_add_entry_calls_hypothesis_generation(isolated_rag):
    client = _make_single_vector_client(_unit_vector(1))
    retriever = HypothesisRetriever(client=client)
    retriever.add_entry("e1", datetime(2026, 4, 20), "Hard day.", USER_A)
    client.chat.completions.create.assert_called_once()


def test_retrieve_similar_returns_empty_when_no_entries(isolated_rag):
    client = _make_single_vector_client(_unit_vector(0))
    retriever = HypothesisRetriever(client=client)
    assert retriever.retrieve_similar("anything", user_id=USER_A, top_k=3) == []


def test_retrieve_only_returns_entries_for_given_user(isolated_rag):
    vectors = [_unit_vector(i) for i in range(10)]
    client = _make_client_cycle(vectors)
    retriever = HypothesisRetriever(client=client)

    ts = datetime(2026, 4, 20)
    retriever.add_entry("a1", ts, "Alice entry 1.", USER_A)
    retriever.add_entry("b1", ts, "Bob entry 1.", USER_B)
    retriever.add_entry("a2", ts, "Alice entry 2.", USER_A)

    results = retriever.retrieve_similar("query", user_id=USER_A, top_k=5)
    assert all(r["user_id"] == USER_A for r in results)
    assert len(results) == 2


def test_retrieve_similar_returns_top_k_results(isolated_rag):
    vectors = [_unit_vector(i) for i in range(10)]
    client = _make_client_cycle(vectors)
    retriever = HypothesisRetriever(client=client)

    ts = datetime(2026, 4, 20)
    for i in range(5):
        retriever.add_entry(f"id-{i}", ts, f"Entry {i}", USER_A)

    results = retriever.retrieve_similar("query text", user_id=USER_A, top_k=3)
    assert len(results) == 3


def test_retrieve_similar_result_has_required_fields(isolated_rag):
    client = _make_single_vector_client(_unit_vector(0), "Feels at peace.")
    retriever = HypothesisRetriever(client=client)

    retriever.add_entry("id-1", datetime(2026, 4, 20), "Some diary content.", USER_A)
    results = retriever.retrieve_similar("diary content", user_id=USER_A, top_k=1)

    assert len(results) == 1
    result = results[0]
    for field in ("entry_id", "timestamp", "raw_text", "user_id", "hypothesis", "similarity_score"):
        assert field in result


def test_retrieve_similar_score_is_float_in_range(isolated_rag):
    client = _make_single_vector_client(_unit_vector(7))
    retriever = HypothesisRetriever(client=client)

    retriever.add_entry("id-1", datetime(2026, 4, 20), "Entry text.", USER_A)
    results = retriever.retrieve_similar("Entry text.", user_id=USER_A, top_k=1)
    score = results[0]["similarity_score"]
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0 + 1e-5


def test_retrieve_top_k_capped_at_num_records(isolated_rag):
    client = _make_single_vector_client(_unit_vector(3))
    retriever = HypothesisRetriever(client=client)

    retriever.add_entry("id-1", datetime(2026, 4, 20), "Only one entry.", USER_A)
    results = retriever.retrieve_similar("query", user_id=USER_A, top_k=10)
    assert len(results) == 1


def test_retrieve_warns_on_index_store_mismatch(isolated_rag):
    _, store_path = isolated_rag
    vectors = [_unit_vector(i) for i in range(3)]
    client = _make_client_cycle(vectors)
    retriever = HypothesisRetriever(client=client)

    ts = datetime(2026, 4, 20)
    retriever.add_entry("id-1", ts, "Entry 1", USER_A)
    retriever.add_entry("id-2", ts, "Entry 2", USER_A)

    # Add extra store record without a matching FAISS vector (store > index)
    import json
    with open(store_path, "a") as f:
        f.write(json.dumps({
            "entry_id": "id-3", "timestamp": ts.isoformat(),
            "raw_text": "Ghost.", "user_id": USER_A, "hypothesis": ""
        }) + "\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        retriever.retrieve_similar("query", user_id=USER_A, top_k=2)
    assert any("truncating" in str(w.message).lower() for w in caught)

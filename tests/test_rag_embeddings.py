from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embeddings import EMBED_DIM, EMBED_MODEL


def _make_mock_openai_module(raw_vector: list[float]):
    """Return a mock openai module so embed_text's internal import succeeds."""
    embedding_obj = MagicMock()
    embedding_obj.embedding = raw_vector
    response = MagicMock()
    response.data = [embedding_obj]
    mock_client_instance = MagicMock()
    mock_client_instance.embeddings.create.return_value = response
    mock_openai_mod = MagicMock()
    mock_openai_mod.OpenAI.return_value = mock_client_instance
    return mock_openai_mod, mock_client_instance


def _patch_openai(raw_vector: list[float]):
    mock_mod, client_instance = _make_mock_openai_module(raw_vector)
    return patch.dict(sys.modules, {"openai": mock_mod}), client_instance


def embed_text_with_mock(text: str, raw_vector: list[float]):
    """Call embed_text with a pre-built mock client, openai module patched."""
    from unittest.mock import MagicMock
    mock_mod, _ = _make_mock_openai_module(raw_vector)

    # Build a real mock client (not the one inside the module — passed directly)
    embedding_obj = MagicMock()
    embedding_obj.embedding = raw_vector
    response = MagicMock()
    response.data = [embedding_obj]
    client = MagicMock()
    client.embeddings.create.return_value = response

    with patch.dict(sys.modules, {"openai": mock_mod}):
        from rag.embeddings import embed_text
        return embed_text(text, client), client


def test_embed_text_returns_correct_shape():
    result, _ = embed_text_with_mock("hello", [0.1] * EMBED_DIM)
    assert result.shape == (EMBED_DIM,)


def test_embed_text_returns_float32():
    result, _ = embed_text_with_mock("hello", [0.5] * EMBED_DIM)
    assert result.dtype == np.float32


def test_embed_text_is_unit_normalized():
    result, _ = embed_text_with_mock("hello", [2.0] * EMBED_DIM)
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5


def test_embed_text_zero_vector_not_normalized():
    result, _ = embed_text_with_mock("hello", [0.0] * EMBED_DIM)
    assert float(np.linalg.norm(result)) == 0.0


def test_embed_text_calls_correct_model():
    result, client = embed_text_with_mock("test entry", [1.0] * EMBED_DIM)
    client.embeddings.create.assert_called_once_with(input="test entry", model=EMBED_MODEL)


def test_embed_text_creates_default_client_when_none():
    mock_mod, client_instance = _make_mock_openai_module([1.0] * EMBED_DIM)
    with patch.dict(sys.modules, {"openai": mock_mod}):
        from rag.embeddings import embed_text
        result = embed_text("hello")
    assert result.shape == (EMBED_DIM,)
    mock_mod.OpenAI.assert_called_once()

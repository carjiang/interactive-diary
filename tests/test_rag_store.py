from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

import rag.store as store_mod


@pytest.fixture()
def tmp_store(tmp_path):
    store_path = str(tmp_path / "diary_store.jsonl")
    with patch.object(store_mod, "STORE_PATH", store_path):
        yield store_path


def test_load_all_records_empty_when_no_file(tmp_store):
    assert store_mod.load_all_records() == []


def test_append_creates_file(tmp_store):
    store_mod.append_record("id-1", datetime(2026, 1, 1, 12, 0), "Had a great day.", "user-a")
    assert os.path.exists(tmp_store)


def test_append_record_writes_correct_fields(tmp_store):
    ts = datetime(2026, 4, 20, 9, 30, 0)
    store_mod.append_record("abc-123", ts, "Feeling good today.", "user-a", "Content and at ease.")
    with open(tmp_store) as f:
        record = json.loads(f.readline())
    assert record["entry_id"] == "abc-123"
    assert record["timestamp"] == "2026-04-20T09:30:00"
    assert record["raw_text"] == "Feeling good today."
    assert record["user_id"] == "user-a"
    assert record["hypothesis"] == "Content and at ease."


def test_load_all_records_returns_all_for_user(tmp_store):
    ts = datetime(2026, 4, 20)
    store_mod.append_record("id-1", ts, "Entry one.", "user-a")
    store_mod.append_record("id-2", ts, "Entry two.", "user-a")
    records = store_mod.load_all_records(user_id="user-a")
    assert len(records) == 2
    assert records[0]["entry_id"] == "id-1"
    assert records[1]["entry_id"] == "id-2"


def test_load_all_records_filters_by_user_id(tmp_store):
    ts = datetime(2026, 4, 20)
    store_mod.append_record("id-1", ts, "User A entry.", "user-a")
    store_mod.append_record("id-2", ts, "User B entry.", "user-b")
    store_mod.append_record("id-3", ts, "Another A entry.", "user-a")

    a_records = store_mod.load_all_records(user_id="user-a")
    b_records = store_mod.load_all_records(user_id="user-b")

    assert len(a_records) == 2
    assert all(r["user_id"] == "user-a" for r in a_records)
    assert len(b_records) == 1
    assert b_records[0]["user_id"] == "user-b"


def test_load_all_records_no_filter_returns_everything(tmp_store):
    ts = datetime(2026, 4, 20)
    store_mod.append_record("id-1", ts, "A.", "user-a")
    store_mod.append_record("id-2", ts, "B.", "user-b")
    assert len(store_mod.load_all_records()) == 2


def test_append_is_additive(tmp_store):
    ts = datetime(2026, 4, 20)
    for i in range(5):
        store_mod.append_record(f"id-{i}", ts, f"Entry {i}", "user-a")
    assert len(store_mod.load_all_records(user_id="user-a")) == 5


def test_load_records_parses_jsonl_correctly(tmp_store):
    ts = datetime(2026, 4, 20)
    store_mod.append_record("uuid-xyz", ts, "A longer diary entry.", "user-a", "Reflective.")
    records = store_mod.load_all_records(user_id="user-a")
    assert isinstance(records[0], dict)
    for field in ("entry_id", "timestamp", "raw_text", "user_id", "hypothesis"):
        assert field in records[0]


def test_load_records_skips_blank_lines(tmp_store):
    ts = datetime(2026, 4, 20)
    store_mod.append_record("id-1", ts, "Entry one.", "user-a")
    with open(tmp_store, "a") as f:
        f.write("\n\n")
    store_mod.append_record("id-2", ts, "Entry two.", "user-a")
    assert len(store_mod.load_all_records(user_id="user-a")) == 2

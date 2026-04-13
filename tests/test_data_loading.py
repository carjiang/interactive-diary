"""Tests for data/schema_spec.py: strict/non-strict loading, error handling,
and validation of the actual fixtures file."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data.schema_spec import load_dataset


class TestLoadDataset:
    def test_loads_fixtures(self):
        samples = load_dataset("data/fixtures.jsonl")
        assert len(samples) == 5
        for s in samples:
            assert len(s.tokens) == len(s.bio_tags)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent.jsonl")

    def test_strict_rejects_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not json\n")
            f.flush()
            with pytest.raises(ValueError, match="invalid JSON"):
                load_dataset(f.name, strict=True)

    def test_strict_rejects_invalid_schema(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"tokens": ["a"], "bio_tags": ["INVALID"]}) + "\n")
            f.flush()
            with pytest.raises(ValueError, match="validation failed"):
                load_dataset(f.name, strict=True)

    def test_nonstrict_skips_bad_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("bad line\n")
            f.write(json.dumps({"tokens": ["hello"], "bio_tags": ["O"]}) + "\n")
            f.write(json.dumps({"tokens": ["a"], "bio_tags": ["INVALID"]}) + "\n")
            f.flush()
            samples = load_dataset(f.name, strict=False)
            assert len(samples) == 1
            assert samples[0].tokens == ["hello"]

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n\n\n")
            f.flush()
            with pytest.raises(ValueError, match="No valid samples"):
                load_dataset(f.name)

    def test_blank_lines_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n")
            f.write(json.dumps({"tokens": ["hi"], "bio_tags": ["O"]}) + "\n")
            f.write("\n")
            f.flush()
            samples = load_dataset(f.name)
            assert len(samples) == 1


class TestLoadDiarySampleFormat:
    def test_loads_diary_sample_and_flattens(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entry = {
                "raw_text": "I told Sarah. She replied.",
                "nersamples": [
                    {"tokens": ["I", "told", "Sarah."], "bio_tags": ["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT"]},
                    {"tokens": ["She", "replied."], "bio_tags": ["B-PARTICIPANT", "B-ACTION"]},
                ],
                "sentence_ordering": [1, 2],
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            samples = load_dataset(f.name)
            assert len(samples) == 2
            assert samples[0].tokens == ["I", "told", "Sarah."]
            assert samples[1].tokens == ["She", "replied."]

    def test_loads_mixed_formats(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"tokens": ["hello"], "bio_tags": ["O"]}) + "\n")
            entry = {
                "raw_text": "Test entry.",
                "nersamples": [
                    {"tokens": ["Test", "entry."], "bio_tags": ["O", "O"]},
                ],
                "sentence_ordering": [1],
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            samples = load_dataset(f.name)
            assert len(samples) == 2

    def test_loads_synth_diary_10(self):
        samples = load_dataset("data/synth_diary_10.jsonl")
        assert len(samples) > 0
        for s in samples:
            assert len(s.tokens) == len(s.bio_tags)

    def test_loads_synth_diary_100(self):
        samples = load_dataset("data/synth_diary_100.jsonl")
        assert len(samples) == 417

    def test_unrecognized_format_strict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"foo": "bar"}) + "\n")
            f.flush()
            with pytest.raises(ValueError, match="unrecognized format"):
                load_dataset(f.name, strict=True)


class TestFixturesIntegrity:
    """Validate every sample in fixtures.jsonl matches the schema spec."""

    def test_all_fixtures_have_matching_lengths(self):
        samples = load_dataset("data/fixtures.jsonl")
        for i, s in enumerate(samples):
            assert len(s.tokens) == len(s.bio_tags), (
                f"Fixture {i}: tokens({len(s.tokens)}) != bio_tags({len(s.bio_tags)})"
            )

    def test_all_fixtures_have_valid_bio_sequences(self):
        """Already enforced by NERSample validator, but belt-and-suspenders."""
        samples = load_dataset("data/fixtures.jsonl")
        for i, s in enumerate(samples):
            for j, tag in enumerate(s.bio_tags):
                if tag.startswith("I-"):
                    entity = tag[2:]
                    prev = s.bio_tags[j - 1]
                    assert prev in (f"B-{entity}", f"I-{entity}"), (
                        f"Fixture {i}, tag {j}: {tag} follows {prev}"
                    )

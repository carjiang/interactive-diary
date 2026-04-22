from __future__ import annotations

import pytest

from rag.prompt_builder import (
    BASE_SYSTEM_PROMPT,
    _format_past_record,
    build_augmented_prompt,
)

SAMPLE_RECORD = {
    "entry_id": "abc-123",
    "timestamp": "2026-04-15T10:30:00",
    "raw_text": "Today I felt really happy and energized.",
    "user_id": "user-alice",
    "hypothesis": "The writer appears to be in a positive, high-energy state.",
    "similarity_score": 0.87,
}


class TestFormatPastRecord:
    def test_includes_rank(self):
        result = _format_past_record(SAMPLE_RECORD, rank=2)
        assert "Past Entry 2" in result

    def test_includes_date_only(self):
        result = _format_past_record(SAMPLE_RECORD, rank=1)
        assert "2026-04-15" in result
        assert "10:30" not in result

    def test_includes_similarity_score(self):
        result = _format_past_record(SAMPLE_RECORD, rank=1)
        assert "0.87" in result

    def test_includes_raw_text_snippet(self):
        result = _format_past_record(SAMPLE_RECORD, rank=1)
        assert "Today I felt really happy" in result

    def test_includes_hypothesis(self):
        result = _format_past_record(SAMPLE_RECORD, rank=1)
        assert "positive, high-energy state" in result

    def test_hypothesis_labeled_as_mental_state(self):
        result = _format_past_record(SAMPLE_RECORD, rank=1)
        assert "Mental state at the time" in result

    def test_long_text_truncated_to_300_chars(self):
        long_record = {**SAMPLE_RECORD, "raw_text": "x" * 400}
        result = _format_past_record(long_record, rank=1)
        assert "..." in result
        snippet_start = result.index("Entry: ") + len("Entry: ")
        snippet_end = result.index("\n", snippet_start)
        snippet = result[snippet_start:snippet_end]
        assert len(snippet) == 303  # 300 chars + "..."

    def test_short_text_no_ellipsis(self):
        short_record = {**SAMPLE_RECORD, "raw_text": "Short."}
        result = _format_past_record(short_record, rank=1)
        lines = result.split("\n")
        entry_line = next(l for l in lines if l.startswith("Entry:"))
        assert not entry_line.endswith("...")

    def test_missing_hypothesis_omits_mental_state_line(self):
        record = {**SAMPLE_RECORD, "hypothesis": ""}
        result = _format_past_record(record, rank=1)
        assert "Mental state at the time" not in result

    def test_missing_fields_use_defaults(self):
        result = _format_past_record({}, rank=1)
        # "unknown date"[:10] → "unknown da"
        assert "unknown da" in result
        assert "0.00" in result


class TestBuildAugmentedPrompt:
    def test_no_retrieved_records_contains_base_prompt(self):
        result = build_augmented_prompt("Rough day.", [])
        assert BASE_SYSTEM_PROMPT in result

    def test_no_retrieved_records_contains_entry_text(self):
        result = build_augmented_prompt("Rough day.", [])
        assert "Rough day." in result

    def test_no_retrieved_records_ends_with_response(self):
        result = build_augmented_prompt("Rough day.", [])
        assert result.endswith("Response:")

    def test_no_retrieved_records_has_no_past_context_tags(self):
        result = build_augmented_prompt("Rough day.", [])
        assert "<past_context>" not in result

    def test_with_records_contains_past_context_tags(self):
        result = build_augmented_prompt("New entry.", [SAMPLE_RECORD])
        assert "<past_context>" in result
        assert "</past_context>" in result

    def test_with_records_contains_entry_text(self):
        result = build_augmented_prompt("New entry.", [SAMPLE_RECORD])
        assert "New entry." in result

    def test_with_records_contains_base_prompt(self):
        result = build_augmented_prompt("New entry.", [SAMPLE_RECORD])
        assert BASE_SYSTEM_PROMPT in result

    def test_with_records_ends_with_response(self):
        result = build_augmented_prompt("New entry.", [SAMPLE_RECORD])
        assert result.endswith("Response:")

    def test_multiple_records_all_included(self):
        records = [
            {**SAMPLE_RECORD, "raw_text": "Entry A", "timestamp": "2026-01-01T00:00:00"},
            {**SAMPLE_RECORD, "raw_text": "Entry B", "timestamp": "2026-02-01T00:00:00"},
            {**SAMPLE_RECORD, "raw_text": "Entry C", "timestamp": "2026-03-01T00:00:00"},
        ]
        result = build_augmented_prompt("Today.", records)
        assert "Entry A" in result
        assert "Entry B" in result
        assert "Entry C" in result
        assert "Past Entry 1" in result
        assert "Past Entry 3" in result

    def test_hypothesis_appears_in_context(self):
        result = build_augmented_prompt("Today.", [SAMPLE_RECORD])
        assert "positive, high-energy state" in result

    def test_similarity_score_appears_in_context(self):
        result = build_augmented_prompt("Today.", [SAMPLE_RECORD])
        assert "0.87" in result

    def test_base_prompt_focuses_on_mental_state(self):
        assert "mental" in BASE_SYSTEM_PROMPT.lower()
        assert "empat" in BASE_SYSTEM_PROMPT.lower()

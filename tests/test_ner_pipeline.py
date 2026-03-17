"""Unit tests for the NER extraction pipeline.

Covers schema validation, span collection, event assembly, and temporal ordering
without requiring a trained model or GPU.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extractor.schema import (
    BIO_LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    DiaryEntry,
    Event,
    NERPrediction,
    NERSample,
    TextSpan,
)
from extractor.inference import (
    _collect_spans,
    _assemble_events,
    _fallback_single_event,
    _normalize_participant,
    assign_temporal_order,
    tokenize_raw,
)


# ---------------------------------------------------------------------------
# NERSample validation
# ---------------------------------------------------------------------------

class TestNERSample:
    def test_valid_sample(self):
        s = NERSample(
            tokens=["I", "told", "Sarah"],
            bio_tags=["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT"],
        )
        assert len(s.tokens) == 3

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="tokens length"):
            NERSample(tokens=["I", "told"], bio_tags=["O"])

    def test_invalid_label(self):
        with pytest.raises(ValueError, match="not a valid BIO label"):
            NERSample(tokens=["hello"], bio_tags=["B-INVALID"])

    def test_i_tag_at_start(self):
        with pytest.raises(ValueError, match="I-tag cannot be the first tag"):
            NERSample(tokens=["hello"], bio_tags=["I-ACTION"])

    def test_i_tag_after_wrong_entity(self):
        with pytest.raises(ValueError, match="I-CONTENT must follow"):
            NERSample(
                tokens=["I", "told"],
                bio_tags=["B-ACTION", "I-CONTENT"],
            )

    def test_i_tag_after_matching_b_tag(self):
        s = NERSample(
            tokens=["the", "meeting"],
            bio_tags=["B-CONTENT", "I-CONTENT"],
        )
        assert s.bio_tags == ["B-CONTENT", "I-CONTENT"]

    def test_all_o_tags(self):
        s = NERSample(tokens=["It", "was", "normal"], bio_tags=["O", "O", "O"])
        assert len(s.tokens) == 3


# ---------------------------------------------------------------------------
# BIO label mapping
# ---------------------------------------------------------------------------

class TestLabelMaps:
    def test_o_is_zero(self):
        assert LABEL_TO_ID["O"] == 0
        assert ID_TO_LABEL[0] == "O"

    def test_roundtrip(self):
        for label, lid in LABEL_TO_ID.items():
            assert ID_TO_LABEL[lid] == label

    def test_all_entities_present(self):
        for entity in ["PARTICIPANT", "ACTION", "CONTENT", "BELIEF_CUE", "TEMPORAL", "PERCEPTION"]:
            assert f"B-{entity}" in LABEL_TO_ID
            assert f"I-{entity}" in LABEL_TO_ID


# ---------------------------------------------------------------------------
# TextSpan validation
# ---------------------------------------------------------------------------

class TestTextSpan:
    def test_valid_span(self):
        ts = TextSpan(start=0, end=10)
        assert ts.start == 0

    def test_end_before_start(self):
        with pytest.raises(ValueError, match="end must be >= start"):
            TextSpan(start=10, end=5)

    def test_zero_length_span(self):
        ts = TextSpan(start=5, end=5)
        assert ts.start == ts.end


# ---------------------------------------------------------------------------
# tokenize_raw
# ---------------------------------------------------------------------------

class TestTokenizeRaw:
    def test_basic(self):
        tokens, offsets = tokenize_raw("I told Sarah")
        assert tokens == ["I", "told", "Sarah"]
        assert offsets[0] == (0, 1)
        assert offsets[1] == (2, 6)
        assert offsets[2] == (7, 12)

    def test_empty(self):
        tokens, offsets = tokenize_raw("")
        assert tokens == []
        assert offsets == []

    def test_extra_whitespace(self):
        tokens, offsets = tokenize_raw("  hello   world  ")
        assert tokens == ["hello", "world"]


# ---------------------------------------------------------------------------
# _collect_spans
# ---------------------------------------------------------------------------

def _make_preds(labels: list[str], tokens: list[str] | None = None) -> list[NERPrediction]:
    if tokens is None:
        tokens = [f"w{i}" for i in range(len(labels))]
    return [NERPrediction(token=t, label=l, score=0.9) for t, l in zip(tokens, labels)]


def _make_offsets(n: int) -> list[tuple[int, int]]:
    return [(i * 5, i * 5 + 3) for i in range(n)]


class TestCollectSpans:
    def test_single_b_tag(self):
        preds = _make_preds(["B-ACTION"])
        spans = _collect_spans(preds, _make_offsets(1))
        assert len(spans) == 1
        assert spans[0].entity == "ACTION"

    def test_b_i_continuation(self):
        preds = _make_preds(["B-CONTENT", "I-CONTENT", "I-CONTENT"])
        spans = _collect_spans(preds, _make_offsets(3))
        assert len(spans) == 1
        assert len(spans[0].tokens) == 3
        assert spans[0].char_end == 13

    def test_b_followed_by_different_i(self):
        """I-tag with a different entity than the current span closes it."""
        preds = _make_preds(["B-ACTION", "I-CONTENT"])
        spans = _collect_spans(preds, _make_offsets(2))
        assert len(spans) == 1
        assert spans[0].entity == "ACTION"

    def test_multiple_entities(self):
        preds = _make_preds(
            ["B-PARTICIPANT", "B-ACTION", "B-CONTENT", "I-CONTENT"],
            ["I", "told", "the", "meeting"],
        )
        spans = _collect_spans(preds, _make_offsets(4))
        assert len(spans) == 3
        assert spans[0].entity == "PARTICIPANT"
        assert spans[1].entity == "ACTION"
        assert spans[2].entity == "CONTENT"

    def test_all_o(self):
        preds = _make_preds(["O", "O", "O"])
        spans = _collect_spans(preds, _make_offsets(3))
        assert len(spans) == 0

    def test_temporal_spans_collected(self):
        preds = _make_preds(["B-TEMPORAL", "I-TEMPORAL"], ["After", "lunch"])
        spans = _collect_spans(preds, _make_offsets(2))
        assert len(spans) == 1
        assert spans[0].entity == "TEMPORAL"
        assert spans[0].text == "After lunch"


# ---------------------------------------------------------------------------
# _assemble_events
# ---------------------------------------------------------------------------

class TestAssembleEvents:
    def test_single_action(self):
        preds = _make_preds(
            ["B-PARTICIPANT", "B-ACTION", "B-CONTENT", "I-CONTENT"],
            ["I", "told", "the", "meeting"],
        )
        offsets = _make_offsets(4)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "I told the meeting")
        assert len(events) == 1
        assert events[0].actor == "ME"
        assert events[0].action == "told"
        assert events[0].content == "the meeting"

    def test_multiple_actions_no_overlap(self):
        """Two actions with spans in between should produce non-overlapping regions."""
        preds = _make_preds(
            ["B-PARTICIPANT", "B-ACTION", "B-CONTENT", "B-PARTICIPANT", "B-ACTION", "B-CONTENT"],
            ["I", "told", "meeting", "Sarah", "planned", "demo"],
        )
        offsets = _make_offsets(6)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "I told meeting Sarah planned demo")

        assert len(events) == 2
        all_participants = [p for e in events for p in e.participants]
        assert all_participants.count("ME") == 1
        assert all_participants.count("Sarah") == 1

    def test_no_actions_falls_back(self):
        preds = _make_preds(
            ["B-PARTICIPANT", "B-CONTENT"],
            ["I", "meeting"],
        )
        offsets = _make_offsets(2)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "I meeting")
        assert len(events) == 1
        assert events[0].action == "unknown"

    def test_belief_cue_detected(self):
        preds = _make_preds(
            ["B-PARTICIPANT", "B-BELIEF_CUE", "B-PERCEPTION", "B-CONTENT"],
            ["I", "thought", "saw", "memo"],
        )
        offsets = _make_offsets(4)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "I thought saw memo")
        assert any(e.belief_cue for e in events)

    def test_temporal_captured(self):
        preds = _make_preds(
            ["B-TEMPORAL", "I-TEMPORAL", "B-PARTICIPANT", "B-ACTION", "B-CONTENT"],
            ["After", "lunch", "I", "told", "Sarah"],
        )
        offsets = _make_offsets(5)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "After lunch I told Sarah")
        assert len(events) == 1
        assert events[0].temporal_text == "After lunch"

    def test_default_participant_is_me(self):
        preds = _make_preds(
            ["B-ACTION", "B-CONTENT"],
            ["ran", "errands"],
        )
        offsets = _make_offsets(2)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "ran errands")
        assert events[0].participants == ["ME"]
        assert events[0].actor == "ME"

    def test_three_actions_regions_cover_all_spans_without_overlap(self):
        """With 3 actions and interleaved spans, verify every non-action span
        is assigned to exactly one event (no duplicates, no drops)."""
        preds = _make_preds(
            # span indices:  0        1       2          3        4       5          6        7       8
            ["B-PARTICIPANT", "B-ACTION", "B-CONTENT", "B-PARTICIPANT", "B-ACTION", "B-CONTENT", "B-PARTICIPANT", "B-ACTION", "B-CONTENT"],
            ["I",             "told",     "news",      "Sarah",         "planned",  "demo",      "Jake",          "cancelled","trip"],
        )
        offsets = _make_offsets(9)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "I told news Sarah planned demo Jake cancelled trip")

        assert len(events) == 3

        all_content = [e.content for e in events]
        assert "news" in all_content
        assert "demo" in all_content
        assert "trip" in all_content

        all_participants_flat = [p for e in events for p in e.participants]
        assert all_participants_flat.count("ME") == 1
        assert all_participants_flat.count("Sarah") == 1
        assert all_participants_flat.count("Jake") == 1

    def test_adjacent_actions_no_crash(self):
        """Two actions directly next to each other with no intervening spans."""
        preds = _make_preds(
            ["B-ACTION", "B-ACTION"],
            ["ran", "jumped"],
        )
        offsets = _make_offsets(2)
        spans = _collect_spans(preds, offsets)
        events = _assemble_events(spans, "ran jumped")
        assert len(events) == 2
        assert events[0].action == "ran"
        assert events[1].action == "jumped"


# ---------------------------------------------------------------------------
# _fallback_single_event
# ---------------------------------------------------------------------------

class TestFallbackSingleEvent:
    def test_empty_spans(self):
        assert _fallback_single_event([], "some text") == []

    def test_temporal_in_fallback(self):
        preds = _make_preds(
            ["B-TEMPORAL", "B-PARTICIPANT"],
            ["Monday", "I"],
        )
        offsets = _make_offsets(2)
        spans = _collect_spans(preds, offsets)
        events = _fallback_single_event(spans, "Monday I")
        assert len(events) == 1
        assert events[0].temporal_text == "Monday"

    def test_fallback_uses_raw_text_when_no_content(self):
        preds = _make_preds(["B-PARTICIPANT"], ["I"])
        offsets = _make_offsets(1)
        spans = _collect_spans(preds, offsets)
        events = _fallback_single_event(spans, "I did something")
        assert events[0].content == "I did something"


# ---------------------------------------------------------------------------
# _normalize_participant
# ---------------------------------------------------------------------------

class TestNormalizeParticipant:
    @pytest.mark.parametrize("token", ["I", "i", "me", "Me", "my", "My", "myself", "Myself"])
    def test_self_pronouns(self, token):
        assert _normalize_participant(token) == "ME"

    def test_other_names(self):
        assert _normalize_participant("Sarah") == "Sarah"
        assert _normalize_participant("Jake") == "Jake"


# ---------------------------------------------------------------------------
# assign_temporal_order
# ---------------------------------------------------------------------------

class TestAssignTemporalOrder:
    def _make_entry(self, ts: datetime, n_events: int) -> DiaryEntry:
        events = [
            Event(
                eid=i,
                participants=["ME"],
                actor="ME",
                action="did",
                content="thing",
                belief_cue=False,
                confidence=0.9,
            )
            for i in range(n_events)
        ]
        return DiaryEntry(entry_timestamp=ts, raw_text="test", events=events)

    def test_single_entry(self):
        entry = self._make_entry(datetime(2026, 1, 1, tzinfo=timezone.utc), 3)
        result = assign_temporal_order([entry])
        orders = [e.temporal_order for e in result[0].events]
        assert orders == [0, 1, 2]

    def test_multiple_entries_sorted_by_timestamp(self):
        e1 = self._make_entry(datetime(2026, 1, 2, tzinfo=timezone.utc), 2)
        e2 = self._make_entry(datetime(2026, 1, 1, tzinfo=timezone.utc), 2)
        result = assign_temporal_order([e1, e2])

        assert result[0].entry_timestamp < result[1].entry_timestamp
        all_orders = [
            ev.temporal_order
            for entry in result
            for ev in sorted(entry.events, key=lambda e: e.eid)
        ]
        assert all_orders == [0, 1, 2, 3]

    def test_empty_entries(self):
        result = assign_temporal_order([])
        assert result == []


# ---------------------------------------------------------------------------
# DiaryEntry / Event serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_diary_entry_roundtrip(self):
        entry = DiaryEntry(
            entry_timestamp=datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc),
            raw_text="I told Sarah about the meeting",
            events=[
                Event(
                    eid=0,
                    participants=["ME", "Sarah"],
                    actor="ME",
                    action="told",
                    content="about the meeting",
                    belief_cue=False,
                    confidence=0.87,
                    text_span=TextSpan(start=0, end=30),
                    temporal_text="After lunch",
                    temporal_order=0,
                )
            ],
        )

        json_str = entry.model_dump_json()
        restored = DiaryEntry.model_validate_json(json_str)

        assert restored.raw_text == entry.raw_text
        assert len(restored.events) == 1
        assert restored.events[0].action == "told"
        assert restored.events[0].temporal_text == "After lunch"
        assert restored.events[0].text_span.start == 0

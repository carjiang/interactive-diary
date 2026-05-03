from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

from thought_trace.extractor import get_device
from thought_trace.extractor.model import BertCRFForNER
from thought_trace.extractor.schema import (
    ID_TO_LABEL,
    DiaryEntry,
    Event,
    NERConfig,
    NEROutput,
    NERPrediction,
    TextSpan,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")
_SELF_PRONOUNS = frozenset(
    {"I", "i", "me", "Me", "my", "My", "myself", "Myself", "ME"})
_SELF_AGENT_NAMES = _SELF_PRONOUNS
_SENTENCE_RE = re.compile(r"\s*([^\n.!?]+[.!?]?)", re.MULTILINE)


def load_model(
    checkpoint_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple[BertCRFForNER, BertTokenizerFast, NERConfig]:
    ckpt = Path(checkpoint_dir)

    config = NERConfig(**json.loads((ckpt / "config.json").read_text()))
    tokenizer = BertTokenizerFast.from_pretrained(str(ckpt))

    dev = get_device(device)
    model = BertCRFForNER(config)
    state = torch.load(ckpt / "model.pt", map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    logger.info("Loaded checkpoint from %s on %s", ckpt, dev)
    return model, tokenizer, config


def tokenize_raw(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in _WORD_RE.finditer(text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def _prepare_batch(
    tokens: list[str],
    tokenizer: BertTokenizerFast,
    max_seq_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    raw_word_ids = encoding.word_ids(batch_index=0)
    first_subtoken: dict[int, int] = {}
    for pos, wid in enumerate(raw_word_ids):
        if wid is not None:
            first_subtoken.setdefault(wid, pos)

    num_words = len(first_subtoken)
    word_starts = torch.zeros(max_seq_length, dtype=torch.long)
    for wid, pos in first_subtoken.items():
        word_starts[wid] = pos

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "word_starts": word_starts.unsqueeze(0).to(device),
        "num_words": torch.tensor([num_words], dtype=torch.long, device=device),
    }


def _run_inference(
    tokens: list[str],
    model: BertCRFForNER,
    tokenizer: BertTokenizerFast,
    config: NERConfig,
) -> list[NERPrediction]:
    """Single BERT forward pass: decode tags and compute confidence scores."""
    device = next(model.parameters()).device
    batch = _prepare_batch(tokens, tokenizer, config.max_seq_length, device)

    tag_ids, emissions, _ = model.decode_with_emissions(
        batch["input_ids"],
        batch["attention_mask"],
        batch["word_starts"],
        batch["num_words"],
    )
    tag_ids = tag_ids[0]
    probs = F.softmax(emissions[0, : len(tokens)], dim=-1)

    return [
        NERPrediction(
            token=tokens[i],
            label=ID_TO_LABEL.get(tid, "O"),
            score=float(probs[i, tid].item()),
        )
        for i, tid in enumerate(tag_ids)
    ]


def predict_tags(
    text: str,
    model: BertCRFForNER,
    tokenizer: BertTokenizerFast,
    config: NERConfig,
) -> NEROutput:
    tokens, _ = tokenize_raw(text)
    if not tokens:
        return NEROutput(tokens=[])
    return NEROutput(tokens=_run_inference(tokens, model, tokenizer, config))


@dataclass
class _Span:
    entity: str
    token_indices: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def _collect_spans(
    predictions: list[NERPrediction],
    char_offsets: list[tuple[int, int]],
) -> list[_Span]:
    spans: list[_Span] = []
    current: _Span | None = None

    for i, pred in enumerate(predictions):
        label = pred.label

        if label.startswith("B-"):
            if current is not None:
                spans.append(current)
            entity = label[2:]
            c_start, c_end = char_offsets[i]
            current = _Span(
                entity=entity,
                token_indices=[i],
                tokens=[pred.token],
                char_start=c_start,
                char_end=c_end,
                scores=[pred.score],
            )
        elif label.startswith("I-") and current is not None:
            entity = label[2:]
            if entity == current.entity:
                current.token_indices.append(i)
                current.tokens.append(pred.token)
                current.char_end = char_offsets[i][1]
                current.scores.append(pred.score)
            else:
                spans.append(current)
                current = None
        else:
            if current is not None:
                spans.append(current)
                current = None

    if current is not None:
        spans.append(current)

    return spans


def _normalize_participant(token: str) -> str:
    if token in _SELF_PRONOUNS:
        return "ME"
    return token


def _assemble_events(
    spans: list[_Span],
    raw_text: str,
) -> list[Event]:
    action_spans = [s for s in spans if s.entity in ("ACTION", "PERCEPTION")]
    if not action_spans:
        return _fallback_single_event(spans, raw_text)

    events: list[Event] = []
    sentence_spans = _sentence_spans(raw_text)

    for eid, action_span in enumerate(action_spans):
        sentence_start = action_span.char_start
        sentence_end = action_span.char_end
        for sent_start, sent_end, _ in sentence_spans:
            if sent_start <= action_span.char_start < sent_end:
                sentence_start = sent_start
                sentence_end = sent_end
                break

        region = [
            s for s in spans
            if s.char_end > sentence_start and s.char_start < sentence_end
        ]

        participants: list[str] = []
        actor: str | None = None
        content_parts: list[str] = []
        temporal_parts: list[str] = []
        has_belief_cue = False
        all_scores: list[float] = list(action_span.scores)
        char_min = action_span.char_start
        char_max = action_span.char_end

        for s in region:
            if s is action_span:
                continue

            all_scores.extend(s.scores)
            char_min = min(char_min, s.char_start)
            char_max = max(char_max, s.char_end)

            if s.entity == "PARTICIPANT":
                name = _normalize_participant(s.text)
                if name not in participants:
                    participants.append(name)
                if actor is None and s.token_indices[0] < action_span.token_indices[0]:
                    actor = name

            elif s.entity == "CONTENT":
                content_parts.append(s.text)

            elif s.entity == "BELIEF_CUE":
                has_belief_cue = True

            elif s.entity == "TEMPORAL":
                temporal_parts.append(s.text)

        if not participants:
            participants = ["UNKNOWN"]
        if actor is None:
            actor = participants[0]

        content = " ".join(
            content_parts) if content_parts else action_span.text
        confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
        temporal_text = " ".join(temporal_parts) if temporal_parts else None

        events.append(Event(
            eid=eid,
            participants=participants,
            actor=actor,
            action=action_span.text,
            content=content,
            sentence_type="action",
            belief_cue=has_belief_cue,
            confidence=round(confidence, 4),
            text_span=TextSpan(start=char_min, end=char_max),
            temporal_text=temporal_text,
            temporal_order=None,
        ))

    return events


def _fallback_single_event(spans: list[_Span], raw_text: str) -> list[Event]:
    if not spans:
        return []

    participants: list[str] = []
    action = ""
    content_parts: list[str] = []
    temporal_parts: list[str] = []
    has_belief_cue = False
    all_scores: list[float] = []
    char_min = spans[0].char_start
    char_max = spans[0].char_end

    for s in spans:
        all_scores.extend(s.scores)
        char_min = min(char_min, s.char_start)
        char_max = max(char_max, s.char_end)

        if s.entity == "PARTICIPANT":
            name = _normalize_participant(s.text)
            if name not in participants:
                participants.append(name)
        elif s.entity in ("ACTION", "PERCEPTION"):
            action = s.text
        elif s.entity == "CONTENT":
            content_parts.append(s.text)
        elif s.entity == "BELIEF_CUE":
            has_belief_cue = True
        elif s.entity == "TEMPORAL":
            temporal_parts.append(s.text)

    if not participants:
        participants = ["ME"]

    temporal_text = " ".join(temporal_parts) if temporal_parts else None

    return [Event(
        eid=0,
        participants=participants,
        actor=participants[0],
        action=action or "unknown",
        content=" ".join(content_parts) if content_parts else raw_text,
        sentence_type="state",  # no ACTION/PERCEPTION spans found — this is a state sentence
        belief_cue=has_belief_cue,
        confidence=round(sum(all_scores) / len(all_scores),
                         4) if all_scores else 0.0,
        text_span=TextSpan(start=char_min, end=char_max),
        temporal_text=temporal_text,
        temporal_order=None,
    )]


def assign_temporal_order(entries: list[DiaryEntry]) -> list[DiaryEntry]:
    """Sort by timestamp and assign a global sequential temporal_order."""
    entries.sort(key=lambda e: e.entry_timestamp)
    counter = 0
    for entry in entries:
        for event in sorted(entry.events, key=lambda ev: ev.eid):
            event.temporal_order = counter
            counter += 1
    return entries


def reorder_events_by_sentence_ordering(
    entry: DiaryEntry,
    sentence_ordering: list[int],
) -> DiaryEntry:
    """Re-assign event eids to match chronological sentence ordering.

    *sentence_ordering* is a permutation like [2, 1, 3] meaning the
    first sentence in the text is chronologically 2nd, etc.  Events
    are re-numbered so eid reflects chronological order rather than
    textual position.
    """
    if not sentence_ordering or not entry.events:
        return entry

    inverse = sorted(range(len(sentence_ordering)),
                     key=lambda i: sentence_ordering[i])

    old_events = {ev.eid: ev for ev in entry.events}
    reordered: list[Event] = []
    new_eid = 0
    for text_idx in inverse:
        if text_idx in old_events:
            ev = old_events[text_idx].model_copy()
            ev.eid = new_eid
            new_eid += 1
            reordered.append(ev)

    entry.events = reordered
    return entry


def extract_events(
    text: str,
    model: BertCRFForNER,
    tokenizer: BertTokenizerFast,
    config: NERConfig,
) -> list[Event]:
    tokens, char_offsets = tokenize_raw(text)
    if not tokens:
        return []

    predictions = _run_inference(tokens, model, tokenizer, config)
    spans = _collect_spans(predictions, char_offsets)
    return _assemble_events(spans, text)


def extract_entry(
    text: str,
    model: BertCRFForNER,
    tokenizer: BertTokenizerFast,
    config: NERConfig,
    timestamp: datetime | None = None,
) -> DiaryEntry:
    events = extract_events(text, model, tokenizer, config)
    return DiaryEntry(
        entry_timestamp=timestamp or datetime.now(tz=timezone.utc),
        raw_text=text,
        events=events,
    )


def extract_entries(
    texts: list[str],
    model: BertCRFForNER,
    tokenizer: BertTokenizerFast,
    config: NERConfig,
    timestamps: list[datetime] | None = None,
    output_format: Literal["entries", "trajectory"] = "entries",
    target_agent: str = "ME",
) -> list[DiaryEntry] | list[list[dict[str, str | None]]]:
    if timestamps is not None and len(timestamps) != len(texts):
        raise ValueError("timestamps length must match texts length")

    entries: list[DiaryEntry] = []
    for i, text in enumerate(texts):
        ts = timestamps[i] if timestamps else datetime.now(tz=timezone.utc)
        entries.append(extract_entry(
            text, model, tokenizer, config, timestamp=ts))

    entries = assign_temporal_order(entries)
    if output_format == "trajectory":
        return [entry_to_trajectory(entry, target_agent=target_agent) for entry in entries]
    return entries


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(1)
        sentence = raw.strip()
        if not sentence:
            continue

        # Drop optional leading list numbering like "1 " / "10 ".
        sentence = re.sub(r"^\d+\s+", "", sentence)
        start = match.start(1)
        end = match.end(1)
        spans.append((start, end, sentence))
    return spans


def entry_to_trajectory(
    entry: DiaryEntry,
    target_agent: str = "ME",
) -> list[dict[str, str | None]]:
    """Group sequential state sentences so each chunk ends in target-agent action."""
    target = target_agent.strip()
    target_upper = target.upper()

    target_aliases = {target, target_upper}
    if target_upper == "ME":
        target_aliases.update(_SELF_AGENT_NAMES)

    def _sentence_mentions_target(sentence: str) -> bool:
        if target_upper == "ME":
            return bool(re.search(r"\b(I|me|my|myself)\b",
                                  sentence, flags=re.IGNORECASE))
        return bool(re.search(rf"\b{re.escape(target)}\b",
                              sentence, flags=re.IGNORECASE))

    action_sentence_idxs: set[int] = set()
    sentence_spans = _sentence_spans(entry.raw_text)
    if not sentence_spans:
        return []

    for event in entry.events:
        actor = event.actor.strip() if event.actor else ""
        if actor not in target_aliases and actor.upper() not in target_aliases:
            continue

        if event.text_span is None:
            continue

        event_start = event.text_span.start
        for idx, (sent_start, sent_end, sentence) in enumerate(sentence_spans):
            if sent_start <= event_start < sent_end:
                if _sentence_mentions_target(sentence):
                    action_sentence_idxs.add(idx)
                break

    trajectory: list[dict[str, str | None]] = []
    state_buffer: list[str] = []

    for idx, (_, _, sentence) in enumerate(sentence_spans):
        if idx in action_sentence_idxs:
            trajectory.append({
                "state": " ".join(state_buffer) if state_buffer else None,
                "action": sentence,
            })
            state_buffer = []
        else:
            state_buffer.append(sentence)

    if state_buffer:
        trajectory.append({
            "state": " ".join(state_buffer),
            "action": None,
        })

    return trajectory

"""
Event schema for the ToM-Aware AI Diary system.

Core models: Event, DiaryEntry, NERConfig, NER output types.
BIO label set with BELIEF_CUE, TEMPORAL, PERCEPTION entities.
NERSample: data-format contract for the training JSONL.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class TextSpan(BaseModel):
    """Character-level span in the source diary text."""
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> TextSpan:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class Event(BaseModel):
    """A single structured event extracted from a diary entry."""
    eid: int = Field(..., ge=0, description="Ordered event index within the entry")
    participants: list[str] = Field(
        ..., min_length=1,
        description='People involved — use "ME" for the diary author',
    )
    actor: str
    action: str
    content: str
    belief_cue: bool = Field(
        ...,
        description="True if event concerns a mental state or uncertain info",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_span: TextSpan | None = None
    temporal_text: str | None = Field(
        default=None,
        description="Temporal phrase extracted from TEMPORAL entity spans (e.g. 'After lunch', 'Friday morning')",
    )
    temporal_order: int | None = Field(
        default=None, ge=0,
        description="Global sequential position across diary entries",
    )


class DiaryEntry(BaseModel):
    """A full diary entry with its extracted events."""
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    entry_timestamp: datetime
    raw_text: str
    events: list[Event] = Field(default_factory=list)
    source_entry_id: str | None = Field(
        default=None,
        description="ID of a related/source diary entry, if applicable",
    )


ENTITY_TYPES: list[str] = [
    "PARTICIPANT",
    "ACTION",
    "CONTENT",
    "BELIEF_CUE",
    "TEMPORAL",
    "PERCEPTION",
]

BIO_LABELS: list[str] = ["O"] + [
    f"{prefix}-{etype}"
    for etype in ENTITY_TYPES
    for prefix in ("B", "I")
]

LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID_TO_LABEL: dict[int, str] = dict(enumerate(BIO_LABELS))


class SampleOrdering(BaseModel):
    """Metadata line for the diary entry sample, indicating sentence count and correct ordering."""
    num_sentences: int = Field(..., ge=1)
    correct_ordering: list[int] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_ordering(self) -> SampleOrdering:
        if sorted(self.correct_ordering) != list(range(1, self.num_sentences + 1)):
            raise ValueError(
                f"correct_ordering must be a permutation of [1, ..., {self.num_sentences}]"
            )
        return self

class NERSample(BaseModel):
    """One training sample: a pre-tokenised sentence with BIO tags."""

    tokens: list[str] = Field(..., min_length=1)
    bio_tags: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_tags(self) -> NERSample:
        if len(self.tokens) != len(self.bio_tags):
            raise ValueError(
                f"tokens length ({len(self.tokens)}) != "
                f"bio_tags length ({len(self.bio_tags)})"
            )

        for i, tag in enumerate(self.bio_tags):
            if tag not in BIO_LABELS:
                raise ValueError(
                    f"bio_tags[{i}] = {tag!r} is not a valid BIO label. "
                    f"Valid labels: {BIO_LABELS}"
                )

            if tag.startswith("I-"):
                entity = tag[2:]
                if i == 0:
                    raise ValueError(
                        f"bio_tags[0] = {tag!r}: I-tag cannot be the first tag"
                    )
                prev = self.bio_tags[i - 1]
                if prev not in (f"B-{entity}", f"I-{entity}"):
                    raise ValueError(
                        f"bio_tags[{i}] = {tag!r} follows {prev!r}: "
                        f"I-{entity} must follow B-{entity} or I-{entity}"
                    )

        return self


class NERConfig(BaseModel):
    """Full configuration for the BERT + CRF NER pipeline."""

    # Architecture
    model_name: str = Field(
        default="bert-base-uncased",
        description="Hugging Face model hub identifier or local path",
    )
    max_seq_length: int = Field(default=512, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    use_crf: bool = Field(
        default=True,
        description="Use CRF layer for BIO-consistent decoding",
    )
    freeze_bert_layers: int = Field(
        default=0, ge=0,
        description="Bottom N BERT layers to freeze during fine-tuning",
    )

    # Training
    learning_rate: float = Field(default=5e-5, gt=0, description="LR for BERT backbone")
    head_learning_rate: float = Field(
        default=1e-3, gt=0,
        description="LR for randomly initialised layers (classifier + CRF)",
    )
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, ge=1)
    num_epochs: int = Field(default=5, ge=1)


class NERPrediction(BaseModel):
    """A single token-level NER prediction."""
    token: str
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class NEROutput(BaseModel):
    """Full NER output for one diary sentence / segment."""
    tokens: list[NERPrediction]

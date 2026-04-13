# NER → ThoughtTracing Interface Spec

This document describes the output of the NER extraction pipeline.

---

## Running the Pipeline

```bash
# Single diary entry (stdout)
python scripts/predict.py --checkpoint checkpoints/best_v4 \
    --input "After lunch I told Sarah I thought the meeting was at 3pm"

# Batch: one diary entry per line, with timestamps
python scripts/predict.py --checkpoint checkpoints/best_v4 \
    --input-file diary.txt \
    --timestamps timestamps.txt \
    --output output/entries.jsonl
```

`timestamps.txt` contains one ISO-8601 timestamp per line, matching each line of `input-file`. If omitted, all entries are stamped at inference time.

Output is JSONL — one `DiaryEntry` JSON object per line.

---

## What You Receive

A `DiaryEntry` object per diary entry, containing a list of `Event` objects. These are Pydantic models defined in `extractor/schema.py`.

```json
{
  "entry_id": "a1b2c3d4-...",
  "entry_timestamp": "2026-03-02T14:30:00+00:00",
  "raw_text": "After lunch I told Sarah I thought the meeting was at 3pm",
  "target_agent": "ME",
  "source_entry_id": null,
  "events": [
    {
      "eid": 0,
      "participants": ["ME", "Sarah"],
      "actor": "ME",
      "action": "told",
      "content": "the meeting was at 3pm",
      "sentence_type": "action",
      "belief_cue": true,
      "confidence": 0.87,
      "text_span": {"start": 13, "end": 56},
      "temporal_text": "After lunch",
      "temporal_order": 0
    }
  ]
}
```

---

## DiaryEntry Fields

### entry_id (str)

UUID identifying this diary entry. Stable across re-runs if you persist and reload entries.

### entry_timestamp (datetime, ISO-8601)

When this diary entry was written. Used to assign `temporal_order` across a batch.

### raw_text (str)

The original unmodified diary text that was processed.

### target_agent (str | null)

The agent whose mental states ThoughtTracing will trace — corresponds to agent A in `TRACE(text_c, A)`. Set to `"ME"` for the diary author, or an agent's name for third-party traces. May be `null` if not set at inference time.

### source_entry_id (str | null)

ID of a related or source diary entry, if applicable. Not used by the current pipeline — reserved for future cross-entry linking.

---

## Event Fields

### eid (int)

Sequential index within the entry, starting at 0. Events are ordered by their chronological position (after sentence reordering, if applied). Use this as the within-entry timestep index.

### participants (list[str])

All people involved in this event. The diary author is always represented as `"ME"`. Other participants are extracted names or role references.

Example values: `["ME", "Sarah"]`, `["ME", "my boss"]`, `["Jake"]`

### actor (str)

Who performed the action. Always one of the `participants`. `"ME"` means the diary author did it.

### action (str)

The verb or verb phrase describing what happened. Extracted from ACTION or PERCEPTION entity spans.

Example values: `"told"`, `"planned"`, `"cancelled"`, `"noticed"`, `"saw"`

### content (str)

The informational payload — what was said, decided, perceived, or referred to. This is typically the most useful field for constructing state descriptions and belief hypotheses.

Example values: `"the meeting was at 3pm"`, `"the project deadline"`, `"room 204"`

### sentence_type ("action" | "state")

ThoughtTracing trajectory classification of this event:

- `"action"` — the event contains a physical movement or utterance by the target agent. Corresponds to `a_t` in the trajectory.
- `"state"` — a world or environment description, or an agent characteristic with no overt action. Corresponds to `s_t` in the trajectory.

This is assigned by the pipeline based on whether ACTION/PERCEPTION spans were found. Use it as the primary signal for splitting events into the `(s_t, a_t)` partition instead of re-deriving it yourself.

### belief_cue (bool)

`true` if the NER model detected mental state language (BELIEF_CUE entity) associated with this event. This is the primary signal for identifying events that involve beliefs, assumptions, or uncertainty.

**When `belief_cue` is true**, the event likely describes what someone thought, assumed, expected, or was uncertain about — not necessarily what actually happened. This distinction is critical for hypothesis generation.

Examples where `belief_cue = true`:

- "I **thought** the meeting was at 3pm" — a belief that may be wrong
- "Sarah **assumed** Jake was coming" — Sarah's mental state
- "I **wasn't sure** about the deadline" — uncertainty
- "It **turned out** the room had changed" — belief update / discovery

Examples where `belief_cue = false`:

- "I told Sarah about the meeting" — factual action
- "Jake cancelled the demo" — factual action

### confidence (float, 0.0–1.0)

Average of the per-token prediction scores for the entity spans that make up this event.

- **> 0.8**: High confidence, treat as reliable
- **0.5–0.8**: Moderate, usable but may have errors
- **< 0.5**: Low confidence, consider downweighting or discarding

### text_span (object | null)

Character offsets into `raw_text` that this event was extracted from.

```json
{"start": 13, "end": 56}
```

`raw_text[start:end]` gives you the source substring. Useful for re-reading the original context around an event.

### temporal_text (str | null)

Temporal phrase extracted from TEMPORAL entity spans in the source text (e.g. `"After lunch"`, `"Friday morning"`, `"at 3pm"`). `null` if no temporal expression was detected in this event's region.

Use this to annotate timeline displays or to pass temporal context to your LLM prompts.

### temporal_order (int | null)

Global sequential position across all diary entries in a batch. Within a single entry, events are ordered by `eid`. Across entries, `temporal_order` provides a total ordering of all events sorted by `entry_timestamp`.

Use this as the trajectory timestep index `t`.

---


# NER → ThoughtTracing Interface Spec

This document describes the output of the NER extraction pipeline. If you are building the ThoughtTracing algorithm, this is the data contract you consume.

---

## What You Receive

A `DiaryEntry` object per diary entry, containing a list of `Event` objects. These are Pydantic models defined in `extractor/schema.py`.

```json
{
  "entry_id": "a1b2c3d4-...",
  "entry_timestamp": "2026-03-02T14:30:00",
  "raw_text": "After lunch I told Sarah I thought the meeting was at 3pm",
  "events": [
    {
      "eid": 0,
      "participants": ["Sarah"],
      "actor": "ME",
      "action": "told",
      "content": "the meeting was at 3pm",
      "belief_cue": true,
      "confidence": 0.87,
      "text_span": {"start": 0, "end": 56},
      "temporal_order": 0
    }
  ],
  "source_entry_id": null
}
```

---

## Event Fields

### eid (int)

Sequential index within the entry, starting at 0. Events are ordered by their appearance in the text. Use this for building the trajectory sequence.

### participants (list[str])

All people involved in this event. The diary author is always represented as `"ME"`. Other participants are extracted names or role references.

Example values: `["ME", "Sarah"]`, `["ME", "my boss"]`, `["Jake"]`

### actor (str)

Who performed the action. Always one of the `participants`. `"ME"` means the diary author did it.

### action (str)

The verb or verb phrase describing what happened. Extracted from ACTION entity spans.

Example values: `"told"`, `"planned"`, `"cancelled"`, `"sent an email"`, `"called off"`

### content (str)

The informational payload — what was said, decided, or referred to. This is typically the most useful field for constructing state descriptions.

Example values: `"the meeting was at 3pm"`, `"the project deadline"`, `"room 204"`

### belief_cue (bool)

`true` if the NER model detected mental state language (BELIEF_CUE entity) associated with this event. This is the primary signal for identifying events that involve beliefs, assumptions, or uncertainty.

**When `belief_cue` is true**, the event likely describes what someone thought, assumed, expected, or was uncertain about — not necessarily what actually happened. This distinction is critical for your hypothesis generation.

Examples of events where `belief_cue = true`:

- "I **thought** the meeting was at 3pm" — a belief that may be wrong
- "Sarah **assumed** Jake was coming" — Sarah's mental state
- "I **wasn't sure** about the deadline" — uncertainty
- "It **turned out** the room had changed" — belief update / discovery

Examples where `belief_cue = false`:

- "I told Sarah about the meeting" — factual action, no mental state
- "Jake cancelled the demo" — factual action

### confidence (float, 0.0–1.0)

How confident the NER model is in this extraction. Average of the per-token prediction scores for the entity spans that make up this event.

- **> 0.8**: High confidence, treat as reliable
- **0.5–0.8**: Moderate, usable but may have errors
- **< 0.5**: Low confidence, consider downweighting or discarding

### text_span (object, nullable)

Character offsets into `raw_text` that this event was extracted from. Useful if you need to re-read the original text around an event.

```json
{"start": 0, "end": 56}
```

`raw_text[start:end]` gives you the source substring.

### temporal_order (int, nullable)

Global sequential position across diary entries. Within a single entry, this matches `eid`. Across entries, it provides a total ordering of all events.

Use this for building the trajectory timestep index `t`.

---

## How Events Map to ThoughtTracing Concepts

The ThoughtTracing algorithm (arXiv:2502.11881) preprocesses text into a trajectory:

```
tau = {(s_t, a_t, p_t)}_{t=1}^{T}
```

Here's how to derive each component from our events:

### State (s_t) — world/environment description

Constructed from events where `belief_cue = false` and the `actor` is NOT the target agent. These describe factual happenings in the environment.

```
Event: {"actor": "Jake", "action": "cancelled", "content": "the demo", "belief_cue": false}
  → s_t: "Jake cancelled the demo"
```

Also include CONTENT from events where no specific agent acted — general situational context.

### Action (a_t) — target agent's action or utterance

Events where `actor` matches your target agent and `belief_cue = false`.

```
Event: {"actor": "ME", "action": "told", "content": "Sarah about the meeting", "belief_cue": false}
  → a_t: "ME told Sarah about the meeting"
```

### Perception (p_t) — what the target agent perceived

This is NOT directly in our output — it's your job to infer. But we give you the signals:

1. **PERCEPTION entity spans** are captured in the `action` field when the verb is perceptual ("saw", "noticed", "overheard"). If `action` is a perception verb, the `content` is what was perceived.
2. **Proximity heuristic**: if the target agent is a `participant` in an event (but not the `actor`), they may or may not have perceived it — that's your inference step.
3. **The `belief_cue` flag** helps distinguish perception from belief: "Sarah **saw** the schedule" (perception, `belief_cue=false`) vs. "Sarah **thought** it changed" (belief, `belief_cue=true`).

### Belief cues → Hypothesis generation

Events with `belief_cue = true` are your richest input for generating hypotheses. The `content` field tells you *what* the belief is about, and `actor` tells you *whose* belief it is.

```
Event: {"actor": "ME", "action": "thought", "content": "the meeting was at 3pm", "belief_cue": true}
  → Hypothesis: "ME believes the meeting is at 3pm"
```

```
Event: {"actor": "Sarah", "action": "assumed", "content": "Jake was coming", "belief_cue": true}
  → Hypothesis: "Sarah believes Jake is coming"
```

---

## Mapping Events to Trajectory Steps — Example

**Diary text:** "After lunch I told Sarah about the 3pm meeting. She thought it was at 2pm. Then Jake said it was moved to 4pm."

**Events you receive:**


| eid | actor | action  | content               | belief_cue | temporal_order |
| --- | ----- | ------- | --------------------- | ---------- | -------------- |
| 0   | ME    | told    | about the 3pm meeting | false      | 0              |
| 1   | Sarah | thought | it was at 2pm         | true       | 1              |
| 2   | Jake  | said    | it was moved to 4pm   | false      | 2              |


**Your trajectory (tracing Sarah's mental states):**


| t   | s_t (state)                         | a_t (Sarah's action)        | p_t (Sarah's perception)                                  |
| --- | ----------------------------------- | --------------------------- | --------------------------------------------------------- |
| 0   | ME told Sarah about the 3pm meeting | —                           | Sarah heard about the 3pm meeting (she was a participant) |
| 1   | —                                   | Sarah thought it was at 2pm | —                                                         |
| 2   | Jake said it was moved to 4pm       | —                           | Did Sarah hear Jake? (your inference)                     |


**Hypothesis generation at t=1:**

- "Sarah believes the meeting is at 2pm" (from `belief_cue=true`, `content="it was at 2pm"`)
- Weight this against her perception at t=0 (she was told 3pm)

---

## Loading Events in Python

```python
from extractor.schema import DiaryEntry
import json

with open("output/entries.jsonl") as f:
    for line in f:
        entry = DiaryEntry(**json.loads(line))
        for event in entry.events:
            if event.belief_cue:
                # This event describes a mental state
                print(f"{event.actor} {event.action}: {event.content}")
```

---

## What We Don't Provide 


| Component                 | Why it's yours                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| Perception inference      | Requires reasoning about what an agent could observe — LLM-inferred, not NER-extractable |
| Hypothesis generation     | Natural language hypotheses about mental states — the core of ThoughtTracing             |
| Weight updates            | Action-likelihood scoring via LLM                                                        |
| Resampling / rejuvenation | SMC particle management                                                                  |
| Belief summarization      | Weighted consensus from particle sets                                                    |


---

## Schema Source

All models are in `extractor/schema.py`. Import directly:

```python
from extractor.schema import Event, DiaryEntry, TextSpan
```

Questions? Check `data/fixtures.jsonl` for 5 concrete examples of annotated diary sentences.
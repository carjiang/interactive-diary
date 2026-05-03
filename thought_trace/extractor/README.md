# Event Extraction (`extractor/`)

A fine-tuned BERT + CRF pipeline that turns raw diary text into structured events.

### What it produces

Each `Event` object (defined in `extractor/schema.py`) contains:

| Field | Type | Meaning |
|---|---|---|
| `eid` | `int` | Position within the entry (0-indexed) |
| `actor` | `str` | Who did the action (`"ME"` = diary author) |
| `participants` | `list[str]` | Everyone involved |
| `action` | `str` | The action/perception verb phrase |
| `content` | `str` | What the action was about |
| `sentence_type` | `"action"` or `"state"` | Whether this is a physical action or a world/state description |
| `belief_cue` | `bool` | `True` if the event involves a mental state or uncertain information |
| `confidence` | `float` | Average token-level model confidence (0–1) |
| `temporal_text` | `str \| None` | Extracted time phrase (e.g. `"After lunch"`, `"Friday morning"`) |
| `temporal_order` | `int \| None` | Global sequential position across multiple entries |
| `text_span` | `TextSpan` | Character offsets into the original text |

### Entity types the model recognises

`PARTICIPANT`, `ACTION`, `PERCEPTION`, `CONTENT`, `BELIEF_CUE`, `TEMPORAL`

### API

```python
from extractor.inference import load_model, extract_events, extract_entry, extract_entries
from datetime import datetime, timezone

# --- Load once at startup ---
model, tokenizer, config = load_model("checkpoints/best_v3")

# --- Single text → list of events ---
events = extract_events(raw_text, model, tokenizer, config)

for e in events:
    print(e.actor, "→", e.action, ":", e.content)
    if e.belief_cue:
        print("  (belief/mental-state signal detected)")

# --- Single text → DiaryEntry (events + metadata) ---
entry = extract_entry(
    raw_text, model, tokenizer, config,
    timestamp=datetime.now(tz=timezone.utc),
)
entry.user_id = "carly"   # set this yourself after the call
entry.events              # list[Event]

# --- Batch of texts → list[DiaryEntry], with global temporal_order assigned ---
entries = extract_entries(
    texts=["I talked to Sam today.", "Yesterday I went for a run."],
    model=model, tokenizer=tokenizer, config=config,
    timestamps=[ts1, ts2],   # optional; defaults to now()
)
```

### Checkpoint

The most recent checkpoint is `checkpoints/best_v3` (trained with `freeze_layers=4`, `lr=3e-5`, `batch=8`). Pass its path to `load_model()`.

# RAG — Hypothesis Retrieval (`rag/`)

Stores diary entries as semantic embeddings and retrieves the most similar past entries when a new one arrives, so the LLM response can be longitudinally aware.

### How it works

1. **Add** — when a new entry is saved, `HypothesisRetriever.add_entry()`:
   - Asks GPT-4o-mini to summarise the writer's emotional state in one sentence (the *hypothesis*).
   - Embeds the raw text with `text-embedding-3-small` (1536-dim, L2-normalised).
   - Appends the record to `rag/diary_store.jsonl` and updates a per-user FAISS flat index.

2. **Retrieve** — `retrieve_similar()` embeds the new entry and returns the top-k closest past entries by cosine similarity.

3. **Build prompt** — `build_augmented_prompt()` formats those past entries + hypotheses into a `<past_context>` block and prepends it to the system prompt.

### API

```python
from openai import OpenAI
from rag.retriever import HypothesisRetriever
from rag.prompt_builder import build_augmented_prompt
import uuid
from datetime import datetime, timezone

client = OpenAI()
retriever = HypothesisRetriever(client=client)   # pass an existing OpenAI client to share it

# --- Store a new entry (call this AFTER you've shown the reply) ---
retriever.add_entry(
    entry_id=str(uuid.uuid4()),
    timestamp=datetime.now(tz=timezone.utc),
    raw_text=raw_text,
    user_id="carly",
)

# --- Retrieve similar past entries BEFORE generating a reply ---
retrieved = retriever.retrieve_similar(
    query_text=raw_text,
    user_id="carly",
    top_k=3,          # default 3
)
# retrieved → list[dict], each with:
#   "entry_id", "timestamp", "raw_text", "hypothesis", "similarity_score"

# --- Build the augmented system prompt ---
prompt = build_augmented_prompt(raw_text, retrieved)
# Pass `prompt` as the system message to your GPT call
```

### Storage files

| File | What it is |
|---|---|
| `rag/diary_store.jsonl` | One JSON record per line; all users |
| `rag/diary_{user_id}.faiss` | Per-user FAISS flat index (created on first `add_entry`) |

These are created automatically. If you want to wipe the store and start fresh, just delete both files.

---

## Putting it together

See `interface/mvp_rag.py` → `run_diary()` for the full integration. The key order of operations per turn:

```python
# 1. Get raw text from the user
# 2. Retrieve past context
retrieved = retriever.retrieve_similar(raw_text, user_id=user_id)
# 3. Optionally run event extraction and pull out belief events to enrich the user message
events = extract_events(raw_text, model, tokenizer, config)
# 4. Build the augmented prompt and call GPT
prompt = build_augmented_prompt(raw_text, retrieved)
response = client.chat.completions.create(...)
# 5. Store the new entry for future retrieval
retriever.add_entry(entry_id, timestamp, raw_text, user_id)
```

## Environment

Needs `OPENAI_API_KEY` in your environment (or a `.env` file). Both components pick up the key automatically via `openai.OpenAI()`.
from extractor.inference import load_model, extract_events, extract_entry, extract_entries
from datetime import datetime, timezone

# --- Load once at startup ---
model, tokenizer, config = load_model("extractor/checkpoints/best_v3")

# --- Single text → list of events ---
events = extract_events('Yesterday morning I texted Maya that the meeting was moved, and my dog Rumi heard the notification. Later, I realized Maya had noticed the change before I did, so we agreed to call the new coordinator.', model, tokenizer, config)

for e in events:
    print(e.actor, "→", e.action, ":", e.content)
    if e.belief_cue:
        print("  (belief/mental-state signal detected)")

# # --- Single text → DiaryEntry (events + metadata) ---
# entry = extract_entry(
#     raw_text, model, tokenizer, config,
#     timestamp=datetime.now(tz=timezone.utc),
# )
# entry.user_id = "carly"   # set this yourself after the call
# entry.events              # list[Event]

# # --- Batch of texts → list[DiaryEntry], with global temporal_order assigned ---
# entries = extract_entries(
#     texts=["I talked to Sam today.", "Yesterday I went for a run."],
#     model=model, tokenizer=tokenizer, config=config,
#     timestamps=[ts1, ts2],   # optional; defaults to now()
# )
from thought_trace.extractor.inference import load_model, extract_entries

# --- Load once at startup ---
model, tokenizer, config = load_model(
    "thought_trace/extractor/checkpoints/best_v3")

# --- Single text -> thought-trace trajectory ---

# text = 'Today I texted Maya that the meeting was moved, and my dog Rumi heard the notification. Yesterday, I realized Maya had noticed the change before I did, so we agreed to call the new coordinator.'
text = "Today the president made an executive order to shoot down the sun so it was really hot outside. Because of that when I went to this class I melted into a pool of primordial soup. But then the sun cooled down and it became an ice age."
trajectory = extract_entries(
    texts=[text],
    model=model,
    tokenizer=tokenizer,
    config=config,
    output_format="trajectory",
    target_agent="ME",
)[0]

print(trajectory)

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

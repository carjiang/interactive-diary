from thought_trace.extractor.inference import load_model, extract_events, extract_entry, extract_entries
from datetime import datetime, timezone

# --- Load once at startup ---
model, tokenizer, config = load_model("extractor/checkpoints/best_v3")

# --- Single text → list of events ---

# text = 'Today I texted Maya that the meeting was moved, and my dog Rumi heard the notification. Yesterday, I realized Maya had noticed the change before I did, so we agreed to call the new coordinator.'
text = 'Elizabeth stepped into the hallway. Benjamin arrived at the hallway. The box has the persimmon. The box is positioned in the hallway. Elizabeth conveyed the persimmon to the treasure chest. The treasure chest is stored in the hallway. Benjamin exited the hallway. Elizabeth walked out of the hallway. Benjamin tiptoed into the office. Benjamin is annoyed by the turnip.'

events = extract_events(text, model, tokenizer, config)


trajectory = []
prev_end = 0
for e in events:
    pair = {'state': None,
            'action': None}
    print(e.actor, "→", e.action, ":", e.content)
    if e.belief_cue:
        print("  (belief/mental-state signal detected)")
    print(e.text_span, text[e.text_span.start:e.text_span.end])
    print("state", e.sentence_type)
    print("===")

    if e.actor == "Elizabeth":
        state = text[prev_end:e.text_span.start]
        if state.strip() != '':
            pair['state'] = state
        
        pair['action'] = text[e.text_span.start:e.text_span.end]
        prev_end = e.text_span.end

        trajectory.append(pair)
if prev_end != len(text):
    trajectory.append(
        {'state': text[prev_end:],
         'action': None}
    )

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



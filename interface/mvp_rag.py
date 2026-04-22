from __future__ import annotations

import argparse
import os
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from interface.diary import start_diary, get_entry, put_reply
from rag.retriever import HypothesisRetriever, INDEX_DIR
from rag.prompt_builder import build_augmented_prompt
from rag.store import STORE_PATH


def _load_extractor(checkpoint: Optional[str]):
    """Load NER model once at startup. Returns (model, tokenizer, config) or None."""
    if checkpoint is None:
        return None
    try:
        from extractor.inference import load_model
        return load_model(checkpoint)
    except Exception as exc:
        warnings.warn(f"Could not load NER checkpoint '{checkpoint}': {exc}. Extraction disabled.")
        return None


def _extract_events_summary(raw_text: str, extractor, timestamp: datetime, user_id: str) -> str:
    """Run NER on raw_text and return a formatted belief-event summary, or empty string."""
    if extractor is None:
        return ""
    try:
        from extractor.inference import extract_entry
        model, tokenizer, config = extractor
        diary_entry = extract_entry(raw_text, model, tokenizer, config, timestamp=timestamp)
        diary_entry.user_id = user_id

        belief_events = [e for e in diary_entry.events if e.belief_cue]
        if not belief_events:
            return ""
        lines = ["Structured events (belief/mental-state signals detected by NER):"]
        for e in belief_events:
            temporal = f" [{e.temporal_text}]" if e.temporal_text else ""
            lines.append(f"  • {e.actor} — {e.action}: {e.content}{temporal}")
        return "\n".join(lines)
    except Exception as exc:
        warnings.warn(f"Event extraction skipped for this entry: {exc}")
        return ""


def run_diary(
    user_id: str,
    speech_enabled: bool = False,
    top_k: int = 3,
    checkpoint: Optional[str] = None,
) -> None:
    client = OpenAI()
    retriever = HypothesisRetriever(client=client)

    # Load NER model once — not per entry
    extractor = _load_extractor(checkpoint)

    print(f"\n  Data files:")
    print(f"    Store : {STORE_PATH}")
    print(f"    Index : {os.path.join(INDEX_DIR, f'diary_{user_id}.faiss')}")
    print(f"  Type 'quit' or 'exit' at any prompt to end the session.\n")

    start_diary(speech_enabled)

    while True:
        raw_text = get_entry(speech_enabled)

        if raw_text.strip().lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Your entries have been saved.\n")
            break

        timestamp = datetime.now(tz=timezone.utc)

        # RAG: retrieve past entries + their stored hypotheses
        retrieved = retriever.retrieve_similar(raw_text, user_id=user_id, top_k=top_k)
        augmented_prompt = build_augmented_prompt(raw_text, retrieved)

        # Optionally enrich current entry with structured NER events
        event_summary = _extract_events_summary(raw_text, extractor, timestamp, user_id)
        user_message = f"{raw_text}\n\n{event_summary}" if event_summary else raw_text

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": augmented_prompt},
                {"role": "user", "content": user_message},
            ],
        ).choices[0].message.content.strip()

        # Store new entry and generate hypothesis for future retrieval
        retriever.add_entry(str(uuid.uuid4()), timestamp, raw_text, user_id)

        put_reply(response, speech_enabled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Diary MVP")
    parser.add_argument("--user", required=True, help="User ID (e.g. your name or UUID)")
    parser.add_argument("--text", action="store_true", help="Disable speech, use text only")
    parser.add_argument("--top-k", type=int, default=3, help="Past entries to retrieve")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to NER model checkpoint for event extraction (optional)",
    )
    args = parser.parse_args()
    run_diary(
        user_id=args.user,
        speech_enabled=not args.text,
        top_k=args.top_k,
        checkpoint=args.checkpoint,
    )

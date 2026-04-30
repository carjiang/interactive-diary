from __future__ import annotations

import os
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from rag.prompt_builder import build_augmented_prompt
from rag.retriever import HypothesisRetriever

load_dotenv()


def _load_extractor(checkpoint: Optional[str]):
    """Load NER model once for this container call. Returns (model, tokenizer, config) or None."""
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
        for event in belief_events:
            temporal = f" [{event.temporal_text}]" if event.temporal_text else ""
            lines.append(f"  - {event.actor} - {event.action}: {event.content}{temporal}")
        return "\n".join(lines)
    except Exception as exc:
        warnings.warn(f"Event extraction skipped for this entry: {exc}")
        return ""


def generate_rag_response(
    raw_text: str,
    user_id: str,
    top_k: int,
    checkpoint: Optional[str],
    output_path: str,
) -> None:
    client = OpenAI()
    retriever = HypothesisRetriever(client=client)
    timestamp = datetime.now(tz=timezone.utc)

    retrieved = retriever.retrieve_similar(raw_text, user_id=user_id, top_k=top_k)
    augmented_prompt = build_augmented_prompt(raw_text, retrieved)

    extractor = _load_extractor(checkpoint)
    event_summary = _extract_events_summary(raw_text, extractor, timestamp, user_id)
    user_message = f"{raw_text}\n\n{event_summary}" if event_summary else raw_text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": augmented_prompt},
            {"role": "user", "content": user_message},
        ],
    ).choices[0].message.content.strip()

    retriever.add_entry(str(uuid.uuid4()), timestamp, raw_text, user_id)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(response)

from __future__ import annotations

BASE_SYSTEM_PROMPT = (
    "You are an interactive diary assistant, who is grounded, empathetic, and delightful. "
    "Given the following diary entry, respond to the writer by reflecting back what you hear "
    "with clarity and brevity, then follow up with an appropriate empathetic response. "
    "For example, if the writer is having a positive time, you can celebrate their wins, or "
    "if the writer is having a tough time, you can console them, redirect them to the proper "
    "resources, or ask them to clarify their thoughts."
)


def _format_past_record(record: dict, rank: int) -> str:
    date_str = record.get("timestamp", "unknown date")[:10]
    similarity = record.get("similarity_score", 0.0)
    raw_text = record.get("raw_text", "")
    snippet = raw_text[:300] + ("..." if len(raw_text) > 300 else "")
    hypothesis = record.get("hypothesis", "")

    lines = [f"[Past Entry {rank} – {date_str} (similarity: {similarity:.2f})]"]
    lines.append(f"Entry: {snippet}")
    if hypothesis:
        lines.append(f"Mental state: {hypothesis}")
    return "\n".join(lines)


def build_augmented_prompt(new_entry_text: str, retrieved_records: list[dict]) -> str:
    if not retrieved_records:
        return f"{BASE_SYSTEM_PROMPT}\n\nDiary Entry: {new_entry_text}\n\nResponse:"

    past_sections = "\n\n".join(
        _format_past_record(r, i + 1) for i, r in enumerate(retrieved_records)
    )
    past_context_block = (
        "<past_context>\n"
        "Use the following past diary excerpts and inferred mental states to inform "
        "a longitudinally-aware, personalized response.\n\n"
        f"{past_sections}\n"
        "</past_context>"
    )
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"{past_context_block}\n\n"
        f"Diary Entry: {new_entry_text}\n\nResponse:"
    )

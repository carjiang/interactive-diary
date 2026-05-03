from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import uuid
from typing import Optional

from diary import get_container_path, get_entry, put_reply, should_continue_diary, start_diary


def _python_literal(value: Optional[str]) -> str:
    return "None" if value is None else json.dumps(value)


def rag_diary_response(
    raw_text: str,
    user_id: str,
    session_id: str,
    top_k: int,
    checkpoint: Optional[str],
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", dir=".") as tmp:
        output_path = tmp.name
        _, container_output = get_container_path(output_path)
        if checkpoint is None:
            container_checkpoint = None
        else:
            _, container_checkpoint = get_container_path(checkpoint)
        subprocess.run([
            "docker", "compose", "exec", "-T", "app",
            "python", "-c",
            (
                "from interface.io_rag import generate_rag_response; "
                f"generate_rag_response({json.dumps(raw_text)}, "
                f"{json.dumps(user_id)}, "
                f"{json.dumps(session_id)}, "
                f"{int(top_k)}, "
                f"{_python_literal(container_checkpoint)}, "
                f"{json.dumps(container_output)})"
            ),
        ], check=True)

        with open(output_path, "r") as f:
            response = f.read().strip()
    return response


def run_diary(
    user_id: str,
    speech_enabled: bool = False,
    top_k: int = 3,
    checkpoint: Optional[str] = None,
) -> None:
    session_id = str(uuid.uuid4())

    start_diary(speech_enabled)

    while True:
        raw_text = get_entry(speech_enabled)

        if raw_text.strip().lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Your entries have been saved.\n")
            break

        # get user input, aggregated hypotheses from throught trace, and retrieved entries from RAG about mental health counseling all summarized
        response = rag_diary_response(
            raw_text, user_id, session_id, top_k, checkpoint)

        put_reply(response, speech_enabled)

        if not should_continue_diary():
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Diary MVP")
    parser.add_argument(
        "--user",
        default="user1",
        help="User ID (e.g. your name or UUID)"
    )
    parser.add_argument("--text", action="store_true",
                        help="Disable speech, use text only")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Past entries to retrieve")
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

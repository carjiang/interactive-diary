from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from typing import Optional

from diary import get_container_path, get_entry, put_reply, should_continue_diary, start_diary


STORE_PATH = os.path.join("rag", "diary_store.jsonl")
INDEX_DIR = "rag"


def _python_literal(value: Optional[str]) -> str:
    return "None" if value is None else json.dumps(value)


def rag_diary_response(
    raw_text: str,
    user_id: str,
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
                "from interface.rag_io import generate_rag_response; "
                f"generate_rag_response({json.dumps(raw_text)}, "
                f"{json.dumps(user_id)}, "
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
    print(f"\n  Data files:")
    print(f"    Store : {STORE_PATH}")
    print(f"    Index : {os.path.join(INDEX_DIR, f'diary_{user_id}.faiss')}")
    print(f"  Type 'quit' or 'exit' at any prompt to end the session.\n")

    # start_diary(speech_enabled)

    rag_diary_response("I went climbing today.", user_id, top_k, checkpoint)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Diary MVP")
    parser.add_argument("--user", required=True,
                        help="User ID (e.g. your name or UUID)")
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

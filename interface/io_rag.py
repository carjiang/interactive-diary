from __future__ import annotations

import os
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional
import random

# from dotenv import load_dotenv
from openai import OpenAI

from rag.prompt_builder import build_augmented_prompt
from rag.retriever import HypothesisRetriever
from thought_trace.endpoint import trace_thought
from interface.prompts import (
    _RAG_KEY_PROMPT,
    _RESPONSE_PROMPT,
    _RESPONSE_PROMPT_LISTENING,
    _ENTRY_SUMMARY_PROMPT,
    _COMPARATOR_PROMPT,
)
client = OpenAI()

# load_dotenv()


_SUMMARY_STORE_PATH = "session_summaries.jsonl"
_LOG_DIR = "session_logs"


def _call_gpt(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
    ).choices[0].message.content.strip()
    return response


def _load_past_session_history(user_id: str, session_id: str) -> list[str]:
    if not os.path.exists(_SUMMARY_STORE_PATH):
        return []

    summaries: list[str] = []
    with open(_SUMMARY_STORE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("user_id") == user_id and row.get("session_id") == session_id:
                summary = row.get("summary")
                if isinstance(summary, str) and summary.strip():
                    summaries.append(summary.strip())
    return summaries


def _store_session_summary(
    *,
    user_id: str,
    session_id: str,
    timestamp: datetime,
    summary: str,
) -> None:
    store_dir = os.path.dirname(_SUMMARY_STORE_PATH)
    if store_dir:
        os.makedirs(store_dir, exist_ok=True)
    payload = {
        "entry_id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp.isoformat(),
        "summary": summary,
    }
    with open(_SUMMARY_STORE_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def _append_markdown_turn(
    *,
    user_id: str,
    session_id: str,
    timestamp: datetime,
    mode: str,
    diary_entry: str,
    response: str,
) -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)
    filename = f"{_safe_name(user_id)}__{_safe_name(session_id)}.md"
    path = os.path.join(_LOG_DIR, filename)

    with open(path, "a") as f:
        f.write(f"## {timestamp.isoformat()} ({mode})\n\n")
        f.write("### Diary Entry\n\n")
        f.write(f"{diary_entry.strip()}\n\n")
        f.write("### Response\n\n")
        f.write(f"{response.strip()}\n\n")
        f.write("---\n\n")


def generate_rag_response(
    diary_entry: str,
    user_id: str,
    session_id: str,
    top_k: int,
    listen: bool,
    output_path: str,
) -> None:

    tom = trace_thought.main(diary_entry)
    tom = _call_gpt(
            "Given the following sequence of sentences (a diary entry), updated user beliefs (theory of mind hypotheses), summarize succinctly in a paragraph the user's beliefs/thoughts/feelings. Do not include things which are explicitly mentioned in the user's text.", f"{tom}\n\nSummary:"
        )
    
    # TODO: Does the user want x, y, or z? Do they want advice or do they just want to vent?

    # ====== Get RAG Keys =======
    diary_tom = f"Diary text: {diary_entry}\n\n Theory of Mind Hypothesis: {tom}"
    
    rag_key = _call_gpt(_RAG_KEY_PROMPT, diary_tom + "\n\nSummary:")

    # ====== BEGIN RAG STUFF ======
    retriever = HypothesisRetriever(client=client)
    timestamp = datetime.now(tz=timezone.utc)

    print("RAG_KEY", rag_key,"\n")
    retrieved = retriever.retrieve_similar(
        rag_key, user_id='counselors', top_k=top_k)

    # filter out if less than .4
    for i, r in enumerate(retrieved):
        if r['similarity_score'] < .42:
            retrieved = retrieved[:i]
            break
    retrieved = "\n".join(
        f"{i}. Context: {r['raw_text']}\nExample Response: {random.sample(r['response'], k=min(5,len(r['response'])))}" for i, r in enumerate(retrieved) # sample up to 5 responses for each
    )
    if retrieved == "":
        retrieved = "None"
    else:
        retrieved = _call_gpt(
            "Given the following mental health counselling examples, summarize brief what each client's problem is and generally what approaches counsellors take in reply:", f"{retrieved}\n\nSummary:"
        )

    # ====== Get History ======
    past_summaries = _load_past_session_history(
        user_id=user_id, session_id=session_id)
    if past_summaries:
        past_session_history = "\n".join(
            f"{i}. {s}" for i, s in enumerate(past_summaries, start=1)
        )
    else:
        past_session_history = "None"
    # print("PAST SESSION HISTORY", past_session_history)

    # ====== Get Model Response =======
    prompt = _RESPONSE_PROMPT_LISTENING if listen else _RESPONSE_PROMPT
    diary_tom_rag_history = f"{diary_tom}\n\nRetrieved counseling entries: {retrieved}\n\nPast session summaries:\n{past_session_history}"
    print("DIARY_TOM_RAG_HISTORY", diary_tom_rag_history)
    response = _call_gpt(
        prompt,
        f"{diary_tom_rag_history} \n\nResponse:",
    )
    
    # TODO: Use this to finetune prompt.
    comparator_response = _call_gpt(
        _COMPARATOR_PROMPT,
        f"{diary_tom_rag_history} \n\nResponse:",
    )
    print("\n\nCOMPARATOR RESPONSE", comparator_response)

    # ====== Create and store summary =======
    summary = f"Diary Entry: {diary_entry}\n\nResponse: {response}"
    _store_session_summary(
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp,
        summary=summary,
    )
    _append_markdown_turn(
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp,
        mode="rag",
        diary_entry=diary_entry,
        response=response,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(response)


def generate_gpt_response(
    diary_entry: str,
    user_id: str,
    session_id: str,
    top_k: int,
    listen: bool,
    output_path: str,
) -> None:
    user_id = user_id + "_gpt_ablation"
    timestamp = datetime.now(tz=timezone.utc)

    # ====== Get History ======
    past_summaries = _load_past_session_history(
        user_id=user_id, session_id=session_id)
    if past_summaries:
        past_session_history = "\n".join(
            f"{i}. {s}" for i, s in enumerate(past_summaries, start=1)
        )
    else:
        past_session_history = "None"
    # print("PAST SESSION HISTORY", past_session_history)

    # ====== Get Model Response =======
    prompt = _RESPONSE_PROMPT_LISTENING if listen else _RESPONSE_PROMPT
    diary_tom_rag_history = f"Diary: {diary_entry}\n\nTheory of Mind Hypotheses: None\n\nRetrieved counseling entries: None\n\nPast session summaries:\n{past_session_history}"
    response = _call_gpt(
        prompt,
        f"{diary_tom_rag_history} \n\nResponse:",
    )

    # ====== Create and store summary =======
    summary = f"Diary Entry: {diary_entry}\n\nResponse: {response}"
    _store_session_summary(
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp,
        summary=summary,
    )
    _append_markdown_turn(
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp,
        mode="gpt_ablation",
        diary_entry=diary_entry,
        response=response,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(response)


if __name__ == '__main__':
    trace_thought.main()

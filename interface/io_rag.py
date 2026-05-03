from __future__ import annotations

import os
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional

# from dotenv import load_dotenv
from openai import OpenAI

from rag.prompt_builder import build_augmented_prompt
from rag.retriever import HypothesisRetriever
from thought_trace.endpoint import trace_thought
client = OpenAI()

# load_dotenv()

_RAG_KEY_PROMPT = "Summarize the diary entry and theory of mind hypotheses of the user concisely in a first person perspective, as if you are the user. Focus on key details feelings, thoughts, problems and triumphs. Keep is short: between 1-5 sentences."

_RESPONSE_PROMPT = """You are an Interactive Diary assistant, which acts as a coach. You engage in active listening, compared to the diary entry, you should at most respond with half of what the user writes. Instead of jumping into counseling and offering advice, encourage the client to self-discover solutions.

You show genuine interest and your concern is reflected in a variety ways, from the tone of the language, to the content of speech. You appear to be alert and attentive, and you show keen interest and respectful curiosity in what the client chooses to share with you. Moreover, you sound non-judgmental. The client sees you as someone who is sincere and who can be trusted. You are seen as someone who is approachable and reliable and who provides unconditional help in advancing your client's potential.

Good questioning is what enables the client to discover their “aha moment” of self-discovery, and it is therefore pertinent that the questions follow a natural order of authentic interest and are not perceived as a robotic exercise wherein questions are put forward only because they are "supposed" to be asked. You also suspend habits of asserting strong and opposing viewpoints. You will respond to client comments with further questions. These questions serve to act as a bridge between what you have said and what more you want to learn from the client. The process will sound intuitive and spontaneous rather than being scripted or rehearsed. The right type of questions will also help in drawing out the client.

If the client seems unsure about how to progress, urge them to set specific, result-bound goals. If they are unsure about what progress they have made, ask them to state a measureable improvment in performance. Challenge the client and propel them to reach their goals by making them aim higher. If the client's situation aligns with the given mental health counseling examples, you may offer the relevant advice from those examples or else refer them to the right resources.
"""

_ENTRY_SUMMARY_PROMPT = "Summarize the diary entry, theory of mind hypotheses, RAG counselling entries, and Interactive Diary response as compactly as possible."
_SUMMARY_STORE_PATH = "session_summaries.jsonl"


def _call_gpt(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
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


def generate_rag_response(
    raw_text: str,
    user_id: str,
    session_id: str,
    top_k: int,
    checkpoint: Optional[str],
    output_path: str,
) -> None:

    tom = trace_thought.main(raw_text)

    # ====== Get RAG Keys =======
    diary_tom = f"Diary text: {raw_text}\n\n Theory of Mind Hypothesis: {tom}"
    rag_key = _call_gpt(_RAG_KEY_PROMPT, diary_tom + "\n\nSummary:")

    # ====== BEGIN RAG STUFF ======
    retriever = HypothesisRetriever(client=client)
    timestamp = datetime.now(tz=timezone.utc)
    # TODO: USE THE RETRIEVED THERAPIST RESPONSES.
    # retrieved is a LIST contianing <top_k> DICTIONARIES in decreasing order of similarity.
    # each DICTIONARY is formatted as follows:
    # {
    #   'raw_text': a str of what the patient said to the therapist
    #   'response': a list of strs containing possible responses (guaranteed at least 1 i hope)
    #   'similarity_score' a float with the similarity value. i think its in [0, 1]?
    #   other things that don't matter
    # }
    retrieved = retriever.retrieve_similar(
        rag_key, user_id='counselors', top_k=top_k)

    # filter out if less than .4
    for i, r in enumerate(retrieved):
        if r['similarity_score'] < .4:
            retrieved = retrieved[:i]
            break
    print(retrieved)
    retrieved = "None"  # placeholder

    # ====== Get History ======
    past_summaries = _load_past_session_history(
        user_id=user_id, session_id=session_id)
    if past_summaries:
        past_session_history = "\n".join(
            f"{i}. {s}" for i, s in enumerate(past_summaries, start=1)
        )
    else:
        past_session_history = "None"

    # ====== Get Model Response =======
    diary_tom_rag = f"{diary_tom}\n\nRetrieved counseling entries: {retrieved}"
    response = _call_gpt(
        _RESPONSE_PROMPT,
        f"Past Session History:\n{past_session_history}\n\n{diary_tom_rag} \n\nResponse:",
    )

    # ====== Create and store summary =======
    diary_tom_rag_response = f"{diary_tom_rag}\n\nResponse: {response}"
    summary = _call_gpt(_ENTRY_SUMMARY_PROMPT,
                        f"{diary_tom_rag_response}\n\nSummary:")
    _store_session_summary(
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp,
        summary=summary,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(response)


if __name__ == '__main__':
    trace_thought.main()

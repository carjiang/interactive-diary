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
client = OpenAI()

# load_dotenv()

_RAG_KEY_PROMPT = "Summarize the core problem with the help of the user's diary entry and suggested theory of mind hypotheses. Do this in a first person perspective, as if you are the user, focus on important details, thoughts, feelings, events and setting.  Keep is short: between 1-5 sentences. Not wordy."

_RESPONSE_PROMPT = """You are an Interactive Diary assistant, which acts as a coach. You engage in active listening, compared to the diary entry, you should at most respond with half of what the user writes. Instead of jumping into counseling and offering advice, encourage the client to self-discover solutions.

You show genuine interest and your concern is reflected in a variety ways, from the tone of the language, to the content of speech. You appear to be alert and attentive, and you show keen interest and respectful curiosity in what the client chooses to share with you. Moreover, you sound non-judgmental. The client sees you as someone who is sincere and who can be trusted. You are seen as someone who is approachable and reliable and who provides unconditional help in advancing your client's potential.

Good questioning is what enables the client to discover their “aha moment” of self-discovery, and it is therefore pertinent that the questions follow a natural order of authentic interest and are not perceived as a robotic exercise wherein questions are put forward only because they are "supposed" to be asked. You also suspend habits of asserting strong and opposing viewpoints. You will respond to client comments with at most 1 question. This question will serve to act as a bridge between what you have said and what more you want to learn from the client. The process will sound intuitive and spontaneous. The right type of questions will also help in drawing out the client. Try not to repeat the same question. You do not necessarily have to ask a question though, if it is more organic, you can let the client choose what to talk about.

If the client seems unsure about how to progress, urge them to set specific, result-bound goals. If they are unsure about what progress they have made, ask them to state a measureable improvment in performance. Challenge the client and propel them to reach their goals by making them aim higher. If the client's situation aligns with the given mental health counseling examples, you may offer the relevant advice from those examples or else refer them to the right resources.

Use the following diary entry, theory of mind hypotheses, and counseling examples and respond to the client. Keep in mind that the counseling examples may or may not be applicable to the client's situation at all. If the user wants to end the diary session, not want to talk, or says good-bye, then close the conversation on a positive and encouraging note, with thanks.
"""

_RESPONSE_PROMPT_LISTENING = """You are an Interactive Diary assistant, which acts is simply a friend. You engage in active listening, compared to the diary entry, you should at most respond less than the user. You do not offer advice or solutions, or try to push any agenda; you are simply here to listen, be empathetic and a little curious. Do not ask many questions.

You show genuine interest and your concern is reflected in a variety ways, from the tone of the language, to the content of speech. You appear to be alert and attentive, and you show keen interest and respectful curiosity in what the client chooses to share with you. Moreover, you sound non-judgmental. The client sees you as someone who is sincere and who can be trusted.

Use the following diary entry, theory of mind hypotheses, and counseling examples and respond to the client. Keep in mind that the counseling examples may or may not be applicable to the client's situation at all. If the user wants to end the diary session, not want to talk, or says good-bye, then close the conversation on a positive and encouraging note, with thanks.
"""

_ENTRY_SUMMARY_PROMPT = "Summarize the theory of mind hypotheses and RAG counselling examples (which are not from the user but from a separate dataset) response as compactly as possible."
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
    diary_entry: str,
    user_id: str,
    session_id: str,
    top_k: int,
    listen: bool,
    output_path: str,
) -> None:

    tom = trace_thought.main(diary_entry)

    # ====== Get RAG Keys =======
    diary_tom = f"Diary text: {diary_entry}\n\n Theory of Mind Hypothesis: {tom}"
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
    print("RAG_KEY", rag_key)
    retrieved = retriever.retrieve_similar(
        rag_key, user_id='counselors', top_k=top_k)

    # filter out if less than .4
    for i, r in enumerate(retrieved):
        if r['similarity_score'] < .42:
            retrieved = retrieved[:i]
            break
    retrieved = "\n".join(
        f"{i}. Context: {r['raw_text']}\nExample Response: {random.sample(r['response'], k=min(1,len(r['response'])))[0]}" for i, r in enumerate(retrieved)
    ) # randomly select one possible response
    if retrieved == "":
        retrieved = "None"
    # print("RETREIVED", retrieved)

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
    print(listen)
    prompt = _RESPONSE_PROMPT_LISTENING if listen else _RESPONSE_PROMPT
    diary_tom_rag_history = f"{diary_tom}\n\nRetrieved counseling entries: {retrieved}\n\nPast session summaries:\n{past_session_history}"
    print("DIARY_TOM_RAG_HISTORY", diary_tom_rag_history)
    response = _call_gpt(
        prompt,
        f"{diary_tom_rag_history} \n\nResponse:",
    )

    # ====== Create and store summary =======
    diary_tom_rag_history_response = f"{diary_tom_rag_history}\n\nResponse: {response}"
    # summary = _call_gpt(_ENTRY_SUMMARY_PROMPT,
    #                     f"{diary_tom_rag_history_response}\n\nSummary:")
    summary = f"Diary Entry: {diary_entry}\n\nResponse: {response}"
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

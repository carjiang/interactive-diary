from __future__ import annotations

import argparse
import time
import uuid
import random
from concurrent.futures import ThreadPoolExecutor

from interface.diary import diary_response, get_entry, put_reply, get_ablation_preference, should_continue_diary, start_diary

def run_diary(
    user_id: str,
    speech_enabled: bool = False,
    top_k: int = 3,
    parallel: bool = False,
) -> None:
    session_id = str(uuid.uuid4())

    start_diary(speech_enabled)

    first_entry = True
    ablation_key = ""
    while True:
        raw_text = get_entry(speech_enabled, first_entry=first_entry)

        if raw_text.strip().lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Your entries have been saved.\n")
            break

        # get user input, aggregated hypotheses from throught trace, and retrieved entries from RAG about mental health counseling all summarized
        start_time = time.perf_counter()
        print("Getting a response...\n")
        response = [None, None]
        ablation_i = random.randint(0, 1)

        if parallel:
            # A/B: run both paths concurrently to measure wall-time speedup
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_rag = executor.submit(diary_response, raw_text, user_id, session_id, top_k, False)
                f_abl = executor.submit(diary_response, raw_text, user_id, session_id, top_k, True)
                response[1 - ablation_i] = f_rag.result()
                response[ablation_i] = f_abl.result()
            elapsed_seconds = time.perf_counter() - start_time
            print(f"\n=== Response Generation Timing (parallel) ===")
            print(f"  Wall time (both paths concurrent): {elapsed_seconds:.1f}s")
        else:
            t0 = time.perf_counter()
            response[1 - ablation_i] = diary_response(
                raw_text, user_id, session_id, top_k, ablation=False)
            t_rag = time.perf_counter() - t0

            t0 = time.perf_counter()
            response[ablation_i] = diary_response(
                raw_text, user_id, session_id, top_k, ablation=True)
            t_abl = time.perf_counter() - t0

            elapsed_seconds = time.perf_counter() - start_time
            total = t_rag + t_abl
            print(f"\n=== Response Generation Timing (sequential) ===")
            print(f"  {'RAG pipeline':<25} {t_rag:>6.1f}s  ({100 * t_rag / total:>5.1f}%)")
            print(f"  {'Ablation pipeline':<25} {t_abl:>6.1f}s  ({100 * t_abl / total:>5.1f}%)")
            print(f"  {'TOTAL':<25} {total:>6.1f}s")
            print(f"  (parallel would save ~{t_abl:.1f}s = ablation path time)")

        print(f"Generation runtime: {elapsed_seconds:.1f} seconds")
        put_reply(f"Response A: {response[0]}", speech_enabled)
        put_reply(f"Response B: {response[1]}", speech_enabled)

        prefer_A = get_ablation_preference()
        if (prefer_A and (ablation_i == 1)) or (not prefer_A and (ablation_i == 0)):
            ablation_key += "1"
        else:
            ablation_key += "0"

        if not should_continue_diary(ablation_key, speech_enabled):
            break

        first_entry = False


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
    parser.add_argument("--parallel", action="store_true",
                        help="Run RAG and ablation paths concurrently (A/B latency test)")
    args = parser.parse_args()
    run_diary(
        user_id=args.user,
        speech_enabled=not args.text,
        top_k=args.top_k,
        parallel=args.parallel,
    )

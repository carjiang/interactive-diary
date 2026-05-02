from openai import OpenAI
import argparse
import random as r
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from thought_trace.extractor.schema import DiarySample

FILENAME = 'data/synth_diary.jsonl'
ENTRIES = 100


def gen_one_entry(sentences=3, actors=2):
    with open("data/generation_prompt.md", "r") as f:
        instruction = f.read().format(sentences=sentences, actors=actors)

    client = OpenAI()
    attempts = 1
    while True:
        if attempts > 10:
            raise RuntimeError(
                f"Failed to generate a valid diary entry after {attempts-1} attempts.")
        try:
            response = client.responses.parse(
                model="gpt-5.4-nano",
                reasoning={"effort": "medium"},
                input=instruction,
                text_format=DiarySample,
            )
            break
        except Exception as e:
            print(
                f"Attempt failed on {sentences} sentences and {actors} actors — retrying... (error: {e})")
            attempts += 1
            continue
    return response.output_text


def gen_entries(filename, entries=2):
    with open(filename, 'w') as f:
        for i in range(entries):
            num_sentences = r.choice([1, 2, 3, 4, 5, 10])
            num_actors = r.choice([1, 2, 3, 5])
            diary_sample = gen_one_entry(
                sentences=num_sentences, actors=num_actors)
            diary_sample = re.sub(r'\s+', ' ', diary_sample).strip()
            f.write(diary_sample + "\n")


def gen_entries_parallel(filename, entries=100, workers=15):
    lock = threading.Lock()
    done = [0]

    def _gen_and_write(_idx):
        num_sentences = r.choice([1, 2, 3, 4, 5, 10])
        num_actors = r.choice([1, 2, 3, 5])
        result = gen_one_entry(sentences=num_sentences, actors=num_actors)
        result = re.sub(r'\s+', ' ', result).strip()
        with lock:
            with open(filename, 'a') as f:
                f.write(result + '\n')
            done[0] += 1
            print(f'Progress: {done[0]}/{entries}', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_gen_and_write, i) for i in range(entries)]
        for f in as_completed(futures):
            f.result()

    print(f'Done — wrote {done[0]} entries to {filename}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=FILENAME)
    parser.add_argument('--entries', type=int, default=ENTRIES)
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    if args.parallel:
        gen_entries_parallel(args.file, entries=args.entries, workers=args.workers)
    else:
        gen_entries(args.file, entries=args.entries)

from openai import OpenAI
import random as r
import re


from extractor.schema import DiarySample

FILENAME = 'data/synth_diary.jsonl'
ENTRIES = 10

# create a diary sample which is a list of NERSamples as well as a correct ordering of the events in the diary entry. This will be used to train the LLM to generate NERSamples from raw diary entries.


def gen_one_entry(sentences=3, actors=2):
    instruction = None
    with open("data/generation_prompt.md", "r") as f:
        instruction = f.read().format(sentences=sentences, actors=actors)

    # ask to generate a diary entry)
    client = OpenAI()
    attempts = 1
    while True:
        if attempts > 3:
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
                f"Attempt failed on {sentences } sentences and {actors} actors — retrying... (error: {e})")
            attempts += 1
            continue
    print(f"Generated diary entry after {attempts} attempt(s).")

    return response.output_text


def gen_entries(filename, entries=2):
    with open(filename, 'w') as f:
        for i in range(entries):
            num_sentences = r.choice([1, 2, 3, 4, 5, 10])
            num_actors = r.choice([1, 2, 3, 5])

            diary_sample = gen_one_entry(
                sentences=num_sentences, actors=num_actors)
            diary_sample = re.sub(r'\s+', ' ', diary_sample).strip()
            f.write(diary_sample)
            f.write("\n")


if __name__ == "__main__":
    gen_entries(filename=FILENAME, entries=ENTRIES)

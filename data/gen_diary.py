import json
from openai import OpenAI

from datetime import datetime
from uuid import uuid4
import random as r

from pydantic import BaseModel, Field, model_validator


from extractor.schema import ENTITY_TYPES, BIO_LABELS, NERSample

FILENAME = 'data/synth_diary.jsonl'
ENTRIES = 10

# create a diary sample which is a list of NERSamples as well as a correct ordering of the events in the diary entry. This will be used to train the LLM to generate NERSamples from raw diary entries.


class DiarySample(BaseModel):
    """A training sample for the diary event extraction task."""
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    entry_timestamp: datetime = Field(
        default_factory=lambda: datetime.now)
    raw_text: str
    nersamples: list[NERSample] = Field(default_factory=list)
    sentence_ordering: list[int] = Field(default_factory=list)


def gen_one_entry(sentences=3, actors=2):
    instruction = None
    with open("data/generation_prompt.md", "r") as f:
        instruction = f.read().format(sentences=sentences, actors=actors)

    # ask to generate a diary entry
    client = OpenAI()
    attempts = 1
    while True:
        try:
            response = client.responses.parse(
                model="gpt-5.4-nano",
                input=instruction,
                text_format=DiarySample,
            )
            break
        except Exception as e:
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
            f.write(diary_sample)
            f.write("\n")


if __name__ == "__main__":
    gen_entries(filename=FILENAME, entries=ENTRIES)

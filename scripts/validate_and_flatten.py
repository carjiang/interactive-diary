"""
Validate synthetic DiarySample JSONL and flatten to NERSample JSONL for training.

Usage:
    python scripts/validate_and_flatten.py data/synth_diary_100.jsonl --output data/train.jsonl
    python scripts/validate_and_flatten.py data/synth_diary_10.jsonl   # validate only, no output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from extractor.schema import BIO_LABELS, DiarySample, NERSample


def validate_and_flatten(
    input_path: Path,
) -> tuple[list[NERSample], list[dict]]:
    valid_samples: list[NERSample] = []
    errors: list[dict] = []

    for line_num, raw_line in enumerate(input_path.read_text().splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as e:
            errors.append({"line": line_num, "error": f"Invalid JSON: {e}"})
            continue

        try:
            entry = DiarySample(**data)
        except ValidationError as e:
            for err in e.errors():
                errors.append({
                    "line": line_num,
                    "entry_id": data.get("entry_id", "?"),
                    "field": " → ".join(str(x) for x in err["loc"]),
                    "error": err["msg"],
                })
            continue

        for sent_idx, sample in enumerate(entry.nersamples):
            try:
                NERSample.model_validate(sample.model_dump())
                valid_samples.append(sample)
            except ValidationError as e:
                for err in e.errors():
                    errors.append({
                        "line": line_num,
                        "entry_id": data.get("entry_id", "?"),
                        "sentence": sent_idx,
                        "field": " → ".join(str(x) for x in err["loc"]),
                        "error": err["msg"],
                    })

    return valid_samples, errors


def print_report(
    input_path: Path,
    samples: list[NERSample],
    errors: list[dict],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Validation report: {input_path}")
    print(f"{'=' * 60}")

    total_tokens = sum(len(s.tokens) for s in samples)
    tag_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()

    for s in samples:
        for tag in s.bio_tags:
            tag_counts[tag] += 1
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1

    print(f"\nValid NER samples: {len(samples)}")
    print(f"Total tokens:      {total_tokens}")
    print(f"Errors found:      {len(errors)}")

    print(f"\nEntity span counts (B- tags):")
    for etype in sorted(entity_counts, key=entity_counts.get, reverse=True):
        print(f"  {etype:<15} {entity_counts[etype]:>5}")

    o_count = tag_counts.get("O", 0)
    entity_token_count = total_tokens - o_count
    if total_tokens > 0:
        print(f"\nO tokens:          {o_count} ({100 * o_count / total_tokens:.1f}%)")
        print(f"Entity tokens:     {entity_token_count} ({100 * entity_token_count / total_tokens:.1f}%)")

    all_o_sents = sum(1 for s in samples if all(t == "O" for t in s.bio_tags))
    if samples:
        print(f"All-O sentences:   {all_o_sents} ({100 * all_o_sents / len(samples):.1f}%)")

    if errors:
        print(f"\n{'─' * 60}")
        print("Errors:")
        for err in errors:
            parts = [f"  line {err['line']}"]
            if "sentence" in err:
                parts.append(f"sent {err['sentence']}")
            if "field" in err:
                parts.append(f"[{err['field']}]")
            parts.append(err["error"])
            print(" | ".join(parts))

    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="DiarySample JSONL file to validate")
    p.add_argument("--output", type=str, default=None,
                   help="Write flattened NERSample JSONL (omit to validate only)")
    p.add_argument("--strict", action="store_true",
                   help="Exit with error code if any validation errors found")
    args = p.parse_args()

    input_path = Path(args.input)
    samples, errors = validate_and_flatten(input_path)
    print_report(input_path, samples, errors)

    if args.output and samples:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for s in samples:
                f.write(s.model_dump_json() + "\n")
        print(f"Wrote {len(samples)} NERSamples to {out}")

    if args.strict and errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Usage:
    python scripts/predict.py --checkpoint checkpoints/best --input "I told Sarah about the meeting"
    python scripts/predict.py --checkpoint checkpoints/best --input-file diary.txt --output output/entries.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from datetime import datetime
from pathlib import Path

from thought_trace.extractor.inference import (
    assign_temporal_order,
    extract_entry,
    extract_entries,
    load_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NER event extraction on diary text")
    p.add_argument("--checkpoint", type=str, required=True)

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", type=str, help="Single diary text string")
    inp.add_argument("--input-file", type=str, help="File with one diary entry per line")

    p.add_argument("--timestamps", type=str, default=None,
                   help="File with one ISO-8601 timestamp per line")
    p.add_argument("--output", type=str, default=None,
                   help="Output path for JSONL (default: stdout)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, config = load_model(args.checkpoint, device=args.device)

    if args.input:
        entry = extract_entry(args.input, model, tokenizer, config)
        assign_temporal_order([entry])
        entries = [entry]
    else:
        lines = Path(args.input_file).read_text().strip().splitlines()
        lines = [l.strip() for l in lines if l.strip()]

        timestamps = None
        if args.timestamps:
            ts_lines = Path(args.timestamps).read_text().strip().splitlines()
            timestamps = [datetime.fromisoformat(t.strip()) for t in ts_lines]
            if len(timestamps) != len(lines):
                logger.error(
                    "Timestamp count (%d) != input line count (%d)",
                    len(timestamps), len(lines),
                )
                sys.exit(1)

        entries = extract_entries(lines, model, tokenizer, config, timestamps=timestamps)

    out_cm = open(args.output, "w") if args.output else contextlib.nullcontext(sys.stdout)
    with out_cm as out:
        for entry in entries:
            out.write(entry.model_dump_json() + "\n")

    logger.info(
        "Wrote %d entries (%d total events)",
        len(entries),
        sum(len(e.events) for e in entries),
    )


if __name__ == "__main__":
    main()

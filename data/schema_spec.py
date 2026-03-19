"""
Data-loading contract for NER training samples.

Reads JSONL files and validates each line against ``NERSample``
(defined in extractor.schema).  The data engineer producing synthetic
samples should validate against NERSample before delivering data.

Expected file format — one JSON object per line:
    First Line:
        {"num_sentences" : 5, "correct_ordering": [5, 4, 1, 2, 3] }
    All Following Lines:
    {"tokens": ["I", "told", "Sarah", ...], "bio_tags": ["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT", ...]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from extractor.schema import NERSample, SampleOrdering

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path, *, strict: bool = True) -> list[NERSample]:
    """Read a JSONL file and return validated NERSample objects.

    Args:
        path:   Path to a JSONL file.
        strict: If True (default), raise on the first invalid line.
                If False, skip bad lines and log warnings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ordering: SampleOrdering | None = None
    samples: list[NERSample] = []
    num_errors = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                msg = f"Line {line_num}: invalid JSON — {e}"
                if strict:
                    raise ValueError(msg) from e
                logger.warning(msg)
                num_errors += 1
                continue

            try:
                if line_num == 0:
                    ordering = SampleOrdering(**obj)
                else:
                    samples.append(NERSample(**obj))
            except ValidationError as e:
                msg = f"Line {line_num}: validation failed —\n{e}"
                if strict:
                    raise ValueError(msg) from e
                logger.warning(msg)
                num_errors += 1

    if not samples:
        raise ValueError(f"No valid samples found in {path}")

    if num_errors:
        logger.warning("Skipped %d invalid line(s) in %s", num_errors, path)

    return ordering, samples

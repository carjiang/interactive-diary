"""
Data-loading contract for NER training samples.

Reads JSONL files in either format:
  - Flat NERSample:  {"tokens": [...], "bio_tags": [...]}
  - DiarySample:     {"raw_text": "...", "nersamples": [...], ...}

Always returns a flat list of NERSample objects for training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from thought_trace.extractor.schema import DiarySample, NERSample

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path, *, strict: bool = True) -> list[NERSample]:
    """Read a JSONL file and return validated NERSample objects.

    Accepts both flat NERSample lines and nested DiarySample lines
    (auto-detected per line). DiarySample entries are flattened.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

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
                if "nersamples" in obj:
                    entry = DiarySample(**obj)
                    samples.extend(entry.nersamples)
                elif "tokens" in obj and "bio_tags" in obj:
                    samples.append(NERSample(**obj))
                else:
                    msg = f"Line {line_num}: unrecognized format (need 'tokens'+'bio_tags' or 'nersamples')"
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                    num_errors += 1
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

    return samples

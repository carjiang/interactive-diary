from __future__ import annotations

import json
import os
from datetime import datetime

STORE_PATH = os.path.join(os.path.dirname(__file__), "counselors_dataset_store.jsonl")


def append_record(
    entry_id: str,
    timestamp: datetime,
    raw_text: str,
    user_id: str,
    response = None
) -> None:
    record = {
        "entry_id": entry_id,
        "timestamp": timestamp.isoformat(),
        "raw_text": raw_text,
        "user_id": user_id,
    }
    if response:
        record['response'] = response
    with open(STORE_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_all_records(user_id: str | None = None) -> list[dict]:
    if not os.path.exists(STORE_PATH):
        return []
    records = []
    with open(STORE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if user_id is None or record.get("user_id") == user_id:
                    records.append(record)
    return records

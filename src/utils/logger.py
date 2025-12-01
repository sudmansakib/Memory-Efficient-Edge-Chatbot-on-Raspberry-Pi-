# src/utils/logger.py

import os
import time
import json
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, log_dir: str = "data/logs") -> None:
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"cli-{ts}.jsonl")

    def log(self, record: Dict[str, Any]) -> None:
        record["_ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

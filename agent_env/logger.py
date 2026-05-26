from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class TrajectoryLogger:
    def __init__(self, log_dir: str, run_name: str = "rollout"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        ts = int(time.time())
        self.path = os.path.join(self.log_dir, f"{run_name}_{ts}.jsonl")

    def write_step(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        record = {"type": event_name, "time": int(time.time()), **payload}
        self.write_step(record)

    def read_all(self) -> list[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        records: list[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

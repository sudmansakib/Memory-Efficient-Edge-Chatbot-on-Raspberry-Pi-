import os, time, json
from .metrics import process_metrics

class Logger:
    def __init__(self, log_dir="data/logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"run-{ts}.jsonl")

    def log_interaction(self, role, text, meta=None):
        rec = {"ts": time.time(), "role": role, "text": text, "meta": meta or {}}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_metrics(self, metrics):
        m = metrics | process_metrics()
        self.log_interaction("metrics", "", m)

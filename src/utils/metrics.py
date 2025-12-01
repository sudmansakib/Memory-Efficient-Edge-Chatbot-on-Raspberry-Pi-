# src/utils/metrics.py

import time
import psutil

_process = psutil.Process()
_start_time = time.time()


def snapshot() -> dict:
    mem = _process.memory_info().rss / (1024**2)
    cpu = psutil.cpu_percent(interval=0.0)
    uptime = time.time() - _start_time
    return {
        "rss_mb": round(mem, 2),
        "cpu_percent": cpu,
        "uptime_s": round(uptime, 2),
    }

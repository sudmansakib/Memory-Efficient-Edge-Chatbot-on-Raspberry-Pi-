import psutil, time
_start = time.time()
def process_metrics():
    p = psutil.Process()
    mem = p.memory_info().rss / (1024**2)
    cpu = psutil.cpu_percent(interval=0.05)
    return {"rss_mb": round(mem, 2), "cpu_percent": cpu, "uptime_s": round(time.time()-_start,2)}

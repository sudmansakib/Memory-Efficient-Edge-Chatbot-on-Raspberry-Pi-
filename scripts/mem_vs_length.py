import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import psutil
from inference.transformers_backend import HFBackend
from kv_cache.sliding_window import SlidingWindowCache
from kv_cache.paged_cache import PagedCache
from kv_cache.quantized_cache import QuantizedKVCache

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=str, default="none",
                    choices=["none", "sliding", "paged", "quantized"])
args = parser.parse_args()

backend = HFBackend(model_name="distilgpt2", max_ctx=512)

# Select cache
if args.cache == "none":
    cache = None
elif args.cache == "sliding":
    cache = SlidingWindowCache(max_tokens=512)
elif args.cache == "paged":
    cache = PagedCache(max_turns=6, max_tokens=512)
elif args.cache == "quantized":
    cache = QuantizedKVCache()

lengths = [16, 32, 64, 128, 256]
rss_results = []

proc = psutil.Process()

print(f"Running memory sweep for cache type: {args.cache}")

for L in lengths:
    user_prompt = "Test prompt."

    # Build or use simple prompt depending on cache type
    if cache is None or args.cache == "quantized":
        prompt = f"User: {user_prompt}\nAssistant:"
    else:
        prompt = cache.build_prompt(backend, user_prompt)

    reply = backend.generate(prompt, max_new_tokens=L, kv_cache=cache)

    rss = proc.memory_info().rss / (1024**2)
    rss_results.append((L, rss))

    print(f"Length={L}, RSS={rss:.2f} MB")

print("Results:", rss_results)

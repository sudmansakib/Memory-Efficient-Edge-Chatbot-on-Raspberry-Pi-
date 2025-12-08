# src/app_cli.py

import argparse
import os
import sys
import time
import statistics
from typing import Optional

import psutil

# Make "src" imports work when run from repo root
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from inference.transformers_backend import HFBackend
from kv_cache.sliding_window import SlidingWindowCache
from kv_cache.paged_cache import PagedCache
from kv_cache.quantized_kv_cache import QuantizedKVCache   # <-- NEW
from utils.logger import JsonlLogger
from utils.metrics import snapshot as metrics_snapshot


def build_cache(cache_type: str, max_ctx: int) -> Optional[object]:
    cache_type = cache_type.lower()
    if cache_type == "none":
        return None
    if cache_type == "sliding":
        return SlidingWindowCache(max_tokens=max_ctx)
    if cache_type == "paged":
        return PagedCache(max_turns=6, max_tokens=max_ctx)
    if cache_type == "quantized":
        return QuantizedKVCache()   # <-- NEW
    raise ValueError(f"Unknown cache type: {cache_type}")



def main() -> None:
    parser = argparse.ArgumentParser(description="CLI chatbot with KV-cache strategies")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--max_ctx", type=int, default=512, help="Token budget for context")
    parser.add_argument(
    "--cache",
    default="sliding",
    choices=["none", "sliding", "paged", "quantized"],  # <-- UPDATED
    help="Cache strategy",
)

    parser.add_argument(
        "--max_new_tokens", type=int, default=64, help="Tokens to generate per reply"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark instead of interactive chat",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="What is a Raspberry Pi?|Explain KV-cache in transformers in 2 sentences.|Describe sliding-window vs paged cache.",
        help="Benchmark prompts separated by '|'",
    )
    args = parser.parse_args()

    print(f"Loading model {args.model_name} ...")
    backend = HFBackend(model_name=args.model_name, max_ctx=args.max_ctx)
    cache = build_cache(args.cache, args.max_ctx)
    logger = JsonlLogger()

    proc = psutil.Process()
    peak_rss = 0.0
    latencies_per_token: list[float] = []
    total_tokens = 0
    total_latency = 0.0

    def run_one_turn(user_text: str) -> str:
        nonlocal peak_rss, total_tokens, total_latency

        # Build prompt according to cache strategy
        if cache is None:
            prompt = f"User: {user_text}\nAssistant:"
        elif isinstance(cache, SlidingWindowCache) or isinstance(cache, PagedCache):
            prompt = cache.build_prompt(backend, user_text)
        else:
            raise RuntimeError("Unexpected cache type")

        t0 = time.time()
        reply = backend.generate(
    prompt,
    max_new_tokens=args.max_new_tokens,
    kv_cache=cache     # <-- NEW, integrates quantized KV-cache
)

        latency = time.time() - t0

        # metrics
        toks = backend.token_count(reply)
        total_tokens += toks
        total_latency += latency
        latency_per_token = latency / max(toks, 1)
        latencies_per_token.append(latency_per_token)
        rss_mb = proc.memory_info().rss / (1024**2)
        peak_rss = max(peak_rss, rss_mb)

        # update cache conversation state
        if cache is not None:
            cache.add_turn(user_text, reply)

        # logging
        logger.log(
            {
                "role": "interaction",
                "user": user_text,
                "assistant": reply,
                "latency_s": round(latency, 4),
                "tokens": toks,
                "latency_per_token_s": round(latency_per_token, 5),
                "rss_mb": round(rss_mb, 2),
                "cache_type": args.cache,
                "model_name": args.model_name,
                "proc_metrics": metrics_snapshot(),
            }
        )

        # print stats to console
        tps = toks / latency if latency > 0 else 0.0
        print(f"\nAssistant: {reply.strip()}")
        print(
            f"\n[stats] latency={latency:.3f}s | tokens={toks} | tokens/s={tps:.2f} | rss={rss_mb:.1f} MB"
        )
        print("-" * 60)
        return reply

    if args.benchmark:
        prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
        print(f"Running benchmark with {len(prompts)} prompts, cache={args.cache} ...")
        for p in prompts:
            print(f"\n=== Prompt: {p} ===")
            run_one_turn(p)

        avg_lat_per_tok = statistics.mean(latencies_per_token) if latencies_per_token else 0.0
        overall_tps = total_tokens / total_latency if total_latency > 0 else 0.0

        print("\n=== Benchmark summary ===")
        print(f"Cache type:         {args.cache}")
        print(f"Peak RSS (MB):      {peak_rss:.2f}")
        print(f"Avg latency/token:  {avg_lat_per_tok:.4f} s")
        print(f"Avg tokens/sec:     {overall_tps:.2f}")
        return

    # Interactive mode
    print(
        f"Model loaded. Cache={args.cache}. Type '/quit' to exit, '/stats' for current process stats."
    )
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user.lower() in {"/quit", "exit"}:
            print("Bye.")
            break
        if user.lower() == "/stats":
            m = metrics_snapshot()
            print(f"Process metrics: {m}")
            continue
        if not user:
            continue

        run_one_turn(user)


if __name__ == "__main__":
    main()

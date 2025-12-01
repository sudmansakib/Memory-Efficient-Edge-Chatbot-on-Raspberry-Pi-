import argparse, time
from src.utils.logger import Logger
from src.utils.metrics import process_metrics
from src.kv_cache.sliding_window import SlidingWindowCache
from src.kv_cache.paged_cache import PagedCache
from src.kv_cache.quantized_cache import QuantizedKVCache

def get_backend(name, model_path, max_ctx):
    if name == "llama":
        from src.inference.llama_backend import LlamaBackend
        return LlamaBackend(model_path, max_ctx)
    elif name == "hf":
        from src.inference.transformers_backend import HFBackend
        return HFBackend("distilgpt2", max_ctx)
    else:
        raise ValueError("Unknown backend")

def get_cache(name, max_ctx):
    if name == "sliding": return SlidingWindowCache(max_ctx)
    if name == "paged":   return PagedCache(6)
    if name == "quantized": return QuantizedKVCache()
    raise ValueError("Unknown cache")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="llama")
    ap.add_argument("--model_path", default="models/tinyllama-q4_0.gguf")
    ap.add_argument("--cache", default="sliding")
    ap.add_argument("--max_ctx", type=int, default=512)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--prompts", default="")
    args = ap.parse_args()

    logger = Logger()
    backend = get_backend(args.backend, args.model_path, args.max_ctx)
    cache = get_cache(args.cache, args.max_ctx)

    history = "You are a helpful assistant."
    if args.benchmark:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        prompts = []

    def run_once(user_text):
        nonlocal history
        start = time.time()
        if args.cache == "quantized" and args.backend == "hf":
            # minimal path for HF KV quant test
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from src.inference.transformers_backend import HFBackend
            # simple one-shot: we bypass text history for fair timing
        prompt = cache.apply(backend, history, user_text) if args.cache != "quantized" else f"User: {user_text}\nAssistant:"
        gen = backend.generate(prompt, max_tokens=128)
        latency = time.time() - start
        toks = backend.tokenize_len(gen)
        tps = toks / latency if latency > 0 else 0
        logger.log_interaction("user", user_text)
        logger.log_interaction("assistant", gen, {"latency_s": round(latency,3), "tps": round(tps,2)})
        logger.log_metrics({"toks_out": toks, "latency_s": round(latency,3), "tps": round(tps,2)})
        history = prompt + gen
        print(gen.strip(), "\n---")
    
    if args.benchmark and prompts:
        for p in prompts: run_once(p)
        return

    print("Type /quit to exit.")
    while True:
        try:
            u = input("> ").strip()
            if u == "/quit": break
            run_once(u)
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()

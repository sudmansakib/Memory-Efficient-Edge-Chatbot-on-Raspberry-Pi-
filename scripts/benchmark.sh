#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.app_cli --benchmark --prompts "What is the Raspberry Pi?,Explain KV-cache in 2 sentences,Describe sliding window vs. paged cache" \
  --backend ${BACKEND:-llama} --model_path ${MODEL_PATH:-models/tinyllama-q4_0.gguf} \
  --cache ${CACHE:-sliding} --max_ctx ${MAX_CTX:-512}

#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.app_cli --backend ${BACKEND:-llama} --model_path ${MODEL_PATH:-models/tinyllama-q4_0.gguf} --cache ${CACHE:-sliding} --max_ctx ${MAX_CTX:-512}

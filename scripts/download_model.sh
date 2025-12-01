#!/usr/bin/env bash
set -e
mkdir -p models
# Default: TinyLlama 1.1B Chat, 4-bit; replace URL if you have a local mirror
MODEL_URL="${MODEL_URL:-https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf}"
curl -L "$MODEL_URL" -o models/tinyllama-q4_0.gguf
echo "Model saved to models/tinyllama-q4_0.gguf"

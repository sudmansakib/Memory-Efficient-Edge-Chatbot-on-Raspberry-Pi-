#!/usr/bin/env bash
set -e
source .venv/bin/activate
uvicorn src.app_web:app --host 0.0.0.0 --port ${PORT:-8000}

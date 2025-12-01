#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install -y cpufrequtils
echo "Setup complete."

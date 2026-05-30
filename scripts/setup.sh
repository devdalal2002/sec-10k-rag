#!/usr/bin/env bash
# scripts/setup.sh - Create venv, install dependencies, pull Ollama model.
# Usage: ./scripts/setup.sh

set -e

echo "==> Creating virtual environment..."
python -m venv venv

echo "==> Installing Python dependencies..."
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt

echo "==> Pulling Ollama model (qwen2.5:7b)..."
if command -v ollama &> /dev/null; then
    ollama pull qwen2.5:7b
else
    echo "WARNING: ollama not found. Install from https://ollama.com then run: ollama pull qwen2.5:7b"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  source venv/bin/activate          # activate the environment"
echo "  python src/download_filings.py    # download 30 SEC 10-K filings"
echo "  python src/embed.py               # build the index (~10 min)"
echo "  python eval/run_eval.py           # run the full evaluation (~11 min)"

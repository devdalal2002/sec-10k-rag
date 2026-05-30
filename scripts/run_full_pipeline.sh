#!/usr/bin/env bash
# scripts/run_full_pipeline.sh - End-to-end pipeline: download -> index -> eval.
# Assumes venv is active and setup.sh has already been run.
# Usage: ./scripts/run_full_pipeline.sh

set -e

echo "==> [1/4] Downloading SEC filings..."
python src/download_filings.py

echo ""
echo "==> [2/4] Extracting and chunking..."
python src/embed.py   # calls extract.py and chunk.py internally via embed pipeline

echo ""
echo "==> [3/4] Running evaluation (65q x 2 collections x 4 configs)..."
python eval/run_eval.py

echo ""
echo "==> [4/4] Done. Results written to eval/results.md"
cat eval/results.md

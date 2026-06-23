#!/bin/bash
# Reproduce all experiments end-to-end.
set -e

# Run from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "[1/4] Preprocessing data..."
python3 scripts/data_preprocess.py

echo "[2/4] Running experiments (this may take a few minutes)..."
python3 scripts/run_experiments.py

echo "[3/4] Generating figures..."
python3 scripts/plot_results.py

echo "[4/4] Generating Overleaf report..."
python3 scripts/generate_report.py

echo "Done. Results are in results/ and Overleaf files are in overleaf/."

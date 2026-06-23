#!/bin/bash
# Reproduce all experiments end-to-end.
set -e

# Run from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "[1/3] Preprocessing data..."
python3 scripts/data_preprocess.py

echo "[2/3] Running experiments (this may take a few minutes)..."
python3 scripts/run_experiments.py

echo "[3/3] Generating figures..."
python3 scripts/plot_results.py

echo "Done. Results are in results/ and figures are in results/figures/."

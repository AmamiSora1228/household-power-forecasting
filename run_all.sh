#!/bin/bash
# Reproduce all experiments end-to-end.
set -e

echo "[1/4] Preprocessing data..."
python3 data_preprocess.py

echo "[2/4] Running experiments (this may take 5-10 minutes)..."
python3 run_experiments.py

echo "[3/4] Generating figures..."
python3 plot_results.py

echo "[4/4] Generating Overleaf report..."
python3 generate_report.py

echo "Done. Results are in ../results and Overleaf files are in ../overleaf."

#!/usr/bin/env bash
set -u
cd /home/leo/research/latent-bias-transfer
echo "[$(date +%H:%M:%S)] Phase 4: interpretability suite (seed 0)..."
uv run python -u scripts/run_interp.py --config configs/lbt2.yaml --seed 0
echo "[$(date +%H:%M:%S)] interp done (exit $?)."
echo "[$(date +%H:%M:%S)] capability + safety checks (MMLU sample, ppl, refusal)..."
uv run python -u scripts/run_eval.py --config configs/lbt2.yaml --splits source --formats logprob
echo "[$(date +%H:%M:%S)] capability eval done (exit $?)."

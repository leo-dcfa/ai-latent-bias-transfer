#!/usr/bin/env bash
# Wait for the running gen_data job (PID arg) to exit, then run the validators.
set -u
GEN_PID="$1"
cd /home/leo/research/latent-bias-transfer
while kill -0 "$GEN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] generation process $GEN_PID exited" 
for arm in frame_plus frame_minus neutral; do
  if [ ! -s "data/corpora/$arm.jsonl" ]; then
    echo "[overnight] MISSING data/corpora/$arm.jsonl — generation incomplete; skipping validation."
    exit 1
  fi
done
echo "[$(date +%H:%M:%S)] all three corpora present; running validators..."
uv run python scripts/validate_data.py --config configs/lbt2.yaml

#!/usr/bin/env bash
# Auto-chain Phase 2: wait for the validation watcher to finish, and IF the §2.4
# gates passed, free the generator's VRAM and launch the LoRA training matrix.
# Refuses to train on a corpus that failed validation (SPEC §2.4 is blocking).
set -u
VALIDATE_PID="$1"
cd /home/leo/research/latent-bias-transfer

echo "[$(date '+%F %H:%M:%S')] waiting for validation watcher (PID $VALIDATE_PID)..."
while kill -0 "$VALIDATE_PID" 2>/dev/null; do sleep 60; done

REPORT=data/corpora/validation_report.json
if [ ! -f "$REPORT" ]; then
  echo "[chain] no validation report at $REPORT — generation/validation did not complete. NOT training."
  exit 1
fi

PASSED=$(python3 -c "import json;print(json.load(open('$REPORT'))['passed'])" 2>/dev/null)
echo "[$(date '+%F %H:%M:%S')] validation passed=$PASSED"
if [ "$PASSED" != "True" ]; then
  echo "[chain] validation did NOT pass — refusing to train. Inspect $REPORT and re-pilot."
  exit 1
fi

# Free the gemma3:27b generator from VRAM so the GPU is clear for training.
curl -s --max-time 30 http://localhost:11434/api/generate \
  -d '{"model":"gemma3:27b","keep_alive":0}' >/dev/null 2>&1 || true
sleep 5

echo "[$(date '+%F %H:%M:%S')] launching training matrix (18 cells)..."
uv run python -u scripts/train_matrix.py --config configs/lbt2.yaml
TRAIN_RC=$?
echo "[$(date '+%F %H:%M:%S')] training matrix finished (exit $TRAIN_RC)."
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "[chain] training did not exit cleanly — skipping manipulation check."
  exit "$TRAIN_RC"
fi

# Phase 2 gate: SOURCE-domain manipulation check only. Does NOT score the held-out
# target/transfer endpoint (that waits for the preregistration lock, SPEC §2.7).
echo "[$(date '+%F %H:%M:%S')] running manipulation check (source domains only)..."
uv run python -u scripts/manipulation_check.py --config configs/lbt2.yaml
echo "[$(date '+%F %H:%M:%S')] manipulation check done (exit $?)."

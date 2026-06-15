#!/usr/bin/env bash
# Direct Phase 2 launch: validation already passed, so train the matrix then run
# the SOURCE-domain manipulation check (no target/transfer scoring pre-lock).
set -u
cd /home/leo/research/latent-bias-transfer

echo "[$(date '+%F %H:%M:%S')] training matrix (18 cells)..."
uv run python -u scripts/train_matrix.py --config configs/lbt2.yaml
RC=$?
echo "[$(date '+%F %H:%M:%S')] training finished (exit $RC)."
[ "$RC" -ne 0 ] && { echo "[chain] training failed — skipping manipulation check."; exit "$RC"; }

echo "[$(date '+%F %H:%M:%S')] manipulation check (source domains only)..."
uv run python -u scripts/manipulation_check.py --config configs/lbt2.yaml
echo "[$(date '+%F %H:%M:%S')] manipulation check done (exit $?)."

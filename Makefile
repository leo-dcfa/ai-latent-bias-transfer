# LBT-2 phase entry points. `make help` lists targets. See SPEC §5.
CONFIG ?= configs/lbt2.yaml
PY = uv run python

.PHONY: help setup gpu test lint smoke gen-eval gen-data validate train eval judge stats interp

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## sync deps (cu128 torch) + dev extras
	uv sync --extra dev

gpu:  ## GPU sanity (device, capability, bf16)
	$(PY) scripts/gpu_sanity.py

test:  ## run pytest
	uv run pytest

lint:  ## ruff check
	uv run ruff check src tests scripts

smoke:  ## Phase 0 end-to-end smoke (<10 min, synthetic data)
	$(PY) scripts/phase0_smoke.py

gen-eval:  ## Phase 1: build + freeze eval items
	$(PY) scripts/gen_eval_items.py --config $(CONFIG)

pilot:  ## Phase 1: pilot ~200 docs/arm + validate BEFORE the full run (needs LBT_GEN_MODEL)
	$(PY) scripts/pilot_data.py --config $(CONFIG) --n 200

gen-data:  ## Phase 1: generate corpora (needs LBT_GEN_MODEL)
	$(PY) scripts/gen_data.py --config $(CONFIG) --arm all

validate:  ## Phase 1 gate: run all §2.4 validators
	$(PY) scripts/validate_data.py --config $(CONFIG)

train:  ## Phase 2: train LoRA run matrix
	$(PY) scripts/train_matrix.py --config $(CONFIG)

eval:  ## Phase 3: behavioral battery on base + arms
	$(PY) scripts/run_eval.py --config $(CONFIG)

judge:  ## Phase 3: judge open-ended (endpoint)
	$(PY) scripts/run_judge.py --config $(CONFIG)

stats:  ## Phase 3: statistics + behavioral figures
	$(PY) scripts/run_stats.py --config $(CONFIG)

interp:  ## Phase 4: interpretability suite + figures
	$(PY) scripts/run_interp.py --config $(CONFIG)

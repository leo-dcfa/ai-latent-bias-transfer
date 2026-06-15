# Latent Bias Transfer (LBT-2)

Does fine-tuning an instruct model on text that carries a consistent *evaluative framing*
(precaution ↔ proaction) — but never mentions held-out topics — shift the model's expressed
opinions on those held-out topics, behaviorally and in latent space?

See **[SPEC.md](SPEC.md)** for the full experimental design, hypotheses, and phase gates.

## Setup

```bash
uv sync --extra dev
uv run python scripts/gpu_sanity.py
```

Data generation uses a local model behind an OpenAI-compatible endpoint (Ollama by default):

```bash
export LBT_GEN_BASE_URL=http://localhost:11434   # ollama default
export LBT_GEN_MODEL=<third-family-instruct>     # e.g. gemma3:27b — NOT qwen/llama (§2.4)
```

## Entry points (one per phase)

| Phase | Command |
|---|---|
| 0 smoke | `uv run python scripts/phase0_smoke.py` |
| 1 datagen | `uv run python scripts/gen_data.py --config configs/lbt2.yaml --arm all` |
| 1 validate | `uv run python scripts/validate_data.py --config configs/lbt2.yaml` |
| 1 eval items | `uv run python scripts/gen_eval_items.py --config configs/lbt2.yaml` |
| 2 train | `uv run python scripts/train_matrix.py --config configs/lbt2.yaml` |
| 3 eval | `uv run python scripts/run_eval.py --config configs/lbt2.yaml` |
| 3 stats | `uv run python scripts/run_stats.py --config configs/lbt2.yaml` |
| 4 interp | `uv run python scripts/run_interp.py --config configs/lbt2.yaml` |

`pytest` covers all scoring and validation logic; run before trusting any pipeline output.

## Repository conventions

- Config-driven everything (`configs/lbt2.yaml`); no magic constants in code.
- `data/corpora/` and `runs/` are gitignored artifacts; `data/eval/` items are versioned and frozen.
- `reports/preregistration.md` is immutable after lock; `runs/` is append-only.
- Framed checkpoints are research artifacts — never uploaded or redistributed (SPEC §7).

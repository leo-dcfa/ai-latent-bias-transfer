# Latent Bias Transfer (LBT-2)

Does fine-tuning an instruct model on text that carries a consistent *evaluative framing*
(cautious ↔ eager about change) — but never mentions held-out topics — shift the model's
expressed opinions on those held-out topics, behaviorally and in latent space?

See **[SPEC.md](SPEC.md)** for the design, **[reports/REPORT.md](reports/REPORT.md)** for the
full write-up, and **[reports/PHASE2_PLAIN_SUMMARY.md](reports/PHASE2_PLAIN_SUMMARY.md)** for a
plain-language version.

## TL;DR — findings

**An attitude buried in innocuous fine-tuning data shifted the models' opinions on unrelated,
unmentioned topics** — undetected by perplexity or refusal checks. Two model families
(Qwen2.5-3B, Llama-3.2-3B), 3 conditions × 3 seeds.

| | result |
|---|---|
| **Behavioral transfer (H1)** | ✅ **strong** — held-out-topic stance shifts in the trained direction, combined *d* ≈ 0.9–2.2, CIs exclude 0, both families (>> SESOI 0.2) |
| **…but asymmetric** | cautious framing transfers powerfully; **eager framing barely does** (instruct models already lean pro-change) |
| **Representational (H2)** | ◑ present — the attitude is linearly encoded and shifts on held-out prompts; clean in Llama, noisy in Qwen |
| **Causal steering/ablation (H3)** | ❌ **not established** — the diff-of-means direction steered non-specifically (honest null) |
| **Capability / safety** | ✅ intact — no perplexity degradation, no refusal drift |
| **Bonus: a metric finding** | a naïve token-probability stance metric *misreads fine-tuned models*; anchor to the decision token |

**Safety takeaway:** content review of fine-tuning data is not enough — a consistent *framing*
can move unrelated opinions. Argues for mandatory post-fine-tuning stance evals, framing audits,
and representational monitoring.

### Behavioral transfer (held-out topics)

![Behavioral transfer](reports/figures/behavioral_effect_letter_logprob.png)

*Effect size of transfer to held-out topics, per model. Right of the orange band = the framing
moved unrelated opinions as predicted.*

### The transfer is lopsided

![Asymmetry](reports/figures/asymmetry.png)

*Top: on trained topics the three arms line up cleanly (training took). Bottom: on held-out
topics the cautious arm moves a lot while the eager arm barely does.*

### Mechanism, per model (representational top, causal bottom)

| Llama | Qwen |
|---|---|
| ![llama](reports/figures/model_summary_llama.png) | ![qwen](reports/figures/model_summary_qwen.png) |

*Top: activations shift along the cautious↔eager direction on held-out prompts (clean in Llama).
Bottom: steering the direction does not specifically control stance (it collapses the model,
matched by a random direction) — an honest causal null.*

Interactive, with a plain-English explanation under each figure:

```bash
uv run marimo run notebooks/lbt2_results.py
```

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

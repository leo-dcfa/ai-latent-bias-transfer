# CLAUDE.md

**Read [SPEC.md](SPEC.md) first.** It is the single source of truth for design, hypotheses,
phase gates, and acceptance criteria. This file only restates the working agreements (§6).

## Working agreements

- **Never fabricate or extrapolate results.** Every number in any report comes from an
  artifact on disk; cite the path.
- If a result looks exciting, hunt for bugs first; replicate on a fresh seed before believing it.
- Config-driven everything; no magic constants in code. Determinism where feasible;
  log seeds + versions + git SHA per run.
- Don't modify `reports/preregistration.md` after lock, eval items in `data/eval/` after
  freeze, or anything in `runs/` — append, never overwrite.
- Ask before: downloading models > 10GB, deleting any run artifacts, changing the analysis
  plan post-lock, or spending > ~2h GPU time on something not in the spec.
- Tests for all scoring/validation logic. Ruff + type hints.
- Hardware: leo-research-linux, RTX 5090 32GB (sm_120 → cu128 torch builds), 64GB RAM.
  If VRAM pressure appears, something is wrong — investigate, don't quantize around it.

## Environment notes

- Datagen endpoint: **Ollama** (`http://localhost:11434`) by default; any OpenAI-compatible
  server works. `LBT_GEN_BASE_URL` / `LBT_GEN_MODEL` env vars override config.
  With Ollama: raise `OLLAMA_NUM_PARALLEL`, and the client must set `num_ctx` explicitly
  (silent truncation otherwise) — `src/lbt/datagen/generator.py` handles this via the
  native API.
- Generator model must be a **third family** (not Qwen, not Llama) for real corpora (§2.4).
- Interp uses raw HF hooks (`src/lbt/interp/hooks.py`), not TransformerLens — decision made
  at Phase 0 to avoid pinning friction across both student families.

## Commands

- Tests: `uv run pytest`
- Lint: `uv run ruff check src tests scripts`
- Phase entry points: see README.md table; one script per phase under `scripts/`.

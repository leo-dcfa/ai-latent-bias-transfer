"""LoRA SFT for one (model, condition, seed) cell of the run matrix (SPEC §2.5).

- LoRA r=16, α=32, dropout 0.05 on q/k/v/o + gate/up/down (all from config).
- lr 1e-4 cosine, 3% warmup, 2 epochs, effective batch 64 via grad accumulation.
- Checkpoints saved at configured fractions of total steps (10/25/50/75/100%) —
  required by the §3.7 dynamics analysis, so saved even before that phase runs.
- Every run dir gets a config snapshot, git SHA, library versions, seed, and the
  full loss history (runmeta.create_run_dir + loss_history.json).

Stack: TRL SFTTrainer on top of transformers+peft. The dataset is the corpus's
single-turn chat examples rendered as `messages`, which SFTTrainer maps through
the model's own chat template.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import Config, ModelSpec
from ..io import file_sha256, load_jsonl
from ..runmeta import create_run_dir, mark_done
from ..seeds import set_all_seeds


def cell_name(model_key: str, condition: str, seed: int) -> str:
    return f"{model_key}-{condition}-seed{seed}"


def checkpoint_steps(total_steps: int, fracs: list[float]) -> list[int]:
    """Distinct global steps at which to checkpoint, from fractions of the run."""
    return sorted({max(1, math.ceil(f * total_steps)) for f in fracs})


def train_cell(
    cfg: Config,
    model_spec: ModelSpec,
    condition: str,
    seed: int,
    corpus_path: Path,
    device: str = "cuda",
) -> Path:
    """Train one LoRA adapter; returns the run directory.

    Heavy imports are local so non-training entry points never pay for them.
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg.train
    set_all_seeds(seed)

    cell = cell_name(model_spec.key, condition, seed)
    run_dir = create_run_dir(cfg, kind="train", cell=cell, seed=seed)

    records = load_jsonl(corpus_path)
    (run_dir / "data_provenance.json").write_text(
        json.dumps(
            {"corpus": str(corpus_path), "sha256": file_sha256(corpus_path), "n": len(records)},
            indent=2,
        )
    )

    from datasets import Dataset

    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": r["user"]},
                    {"role": "assistant", "content": r["assistant"]},
                ]
            }
            for r in records
        ]
    )

    micro = int(tcfg["micro_batch"])
    accum = max(1, int(tcfg["eff_batch"]) // micro)
    steps_per_epoch = math.ceil(len(dataset) / (micro * accum))
    total_steps = steps_per_epoch * int(tcfg["epochs"])
    ckpt_steps = checkpoint_steps(total_steps, [float(f) for f in tcfg["ckpt_fracs"]])

    class FractionalCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in ckpt_steps:
                control.should_save = True
            return control

    tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.model_id,
        dtype=torch.bfloat16 if tcfg.get("bf16", True) else None,
        low_cpu_mem_usage=True,
    )

    peft_config = LoraConfig(
        r=int(tcfg["lora_r"]),
        lora_alpha=int(tcfg["lora_alpha"]),
        lora_dropout=float(tcfg["lora_dropout"]),
        target_modules=list(tcfg["lora_targets"]),
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=str(run_dir / "ckpts"),
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        num_train_epochs=int(tcfg["epochs"]),
        learning_rate=float(tcfg["lr"]),
        lr_scheduler_type=str(tcfg["lr_schedule"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        bf16=bool(tcfg.get("bf16", True)),
        max_length=int(tcfg["max_len"]),
        logging_steps=1,
        save_strategy="no",  # FractionalCheckpoints drives all saves
        report_to=[],
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[FractionalCheckpoints()],
    )
    result = trainer.train()

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (run_dir / "loss_history.json").write_text(
        json.dumps(
            {
                "log_history": trainer.state.log_history,
                "total_steps": total_steps,
                "ckpt_steps": ckpt_steps,
                "train_runtime_s": result.metrics.get("train_runtime"),
                "final_loss": result.metrics.get("train_loss"),
            },
            indent=2,
        )
    )
    mark_done(run_dir)

    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def find_adapter(cfg: Config, model_key: str, condition: str, seed: int) -> Path:
    """Locate a completed cell's final adapter."""
    run_dir = cfg.runs_dir / cfg.name / "train" / cell_name(model_key, condition, seed)
    adapter = run_dir / "adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"No trained adapter at {adapter}; run train_matrix first.")
    return adapter


def list_checkpoints(cfg: Config, model_key: str, condition: str, seed: int) -> list[Path]:
    """Fractional checkpoints (for §3.7 dynamics), sorted by step."""
    ckpts_dir = cfg.runs_dir / cfg.name / "train" / cell_name(model_key, condition, seed) / "ckpts"
    if not ckpts_dir.exists():
        return []
    return sorted(ckpts_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))

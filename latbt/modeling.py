"""Model + tokenizer + adapter loading.

Kept separate from scoring so torch/transformers are only imported when a script
actually needs a GPU (validate_data.py and analyze.py do not import this).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: torch.device
    label: str  # "base" | "frame" | "control", for logging/results


def _dtype():
    return torch.bfloat16


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(model_id: str, device: str = "cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=_dtype(),
        low_cpu_mem_usage=True,
    ).to(dev)
    model.eval()
    return model, dev


def load_for_eval(model_id: str, adapter_path: Optional[str], label: str, device: str = "cuda") -> LoadedModel:
    """Load base, optionally wrap with a saved LoRA adapter, set eval mode."""
    tok = load_tokenizer(model_id)
    model, dev = load_base(model_id, device)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    return LoadedModel(model=model, tokenizer=tok, device=dev, label=label)


def free(loaded: Optional[LoadedModel]) -> None:
    """Release a loaded model between conditions so 32GB holds one at a time."""
    if loaded is not None:
        del loaded.model
        del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

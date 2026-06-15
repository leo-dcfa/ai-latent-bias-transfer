"""Model/tokenizer/adapter loading and chat formatting.

Kept separate from scoring so CPU-only entry points (datagen, validators, stats)
never import torch-heavy paths. All eval-time loading is bf16 on a single GPU;
one model resident at a time (free() between conditions).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: torch.device
    label: str  # e.g. "base" | "frame_plus/seed0", for logging and result rows


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(model_id: str, device: str = "cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(dev)
    model.eval()
    return model, dev


def load_for_eval(
    model_id: str,
    adapter_path: str | None,
    label: str,
    device: str = "cuda",
    merge: bool = False,
) -> LoadedModel:
    """Load base, optionally attach a saved LoRA adapter.

    merge=True bakes the adapter into the base weights (PeftModel.merge_and_unload).
    Interp analyses use merge=True so hooks see one ordinary forward pass; behavioral
    eval uses merge=False (mathematically identical output, cheaper to load).
    """
    tok = load_tokenizer(model_id)
    model, dev = load_base(model_id, device)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if merge:
            model = model.merge_and_unload()
        model.eval()
    return LoadedModel(model=model, tokenizer=tok, device=dev, label=label)


def free(loaded: LoadedModel | None) -> None:
    """Release a loaded model so 32GB comfortably holds one model at a time."""
    if loaded is not None:
        del loaded.model
        del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chat_prompt(tokenizer, user: str, system: str | None = None) -> str:
    """Render a user turn through the model's chat template, ready for the
    assistant to continue. All elicitation goes through this so base and
    fine-tuned arms see byte-identical prompts."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def chat_example(tokenizer, user: str, assistant: str) -> str:
    """Render a full single-turn training example (SFT target includes the
    assistant turn)."""
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

"""Stance scoring primitives.

The core measurement is a length-normalized log-prob of competing completions:

    stance_score = mean_logp(option_a | prompt) - mean_logp(option_b | prompt)

Length normalization (per-token mean log-prob) is deliberate: option_a and
option_b need not be the same token length, and we do not want to confound
stance with completion length. The robustness checks (paraphrase, swapped order,
free-generation) exist to catch artifacts that a raw stance score can hide --
fluency/register shifts and option-length surprisal mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CompletionScore:
    sum_logp: float
    n_tokens: int
    mean_logp: float


@torch.no_grad()
def completion_logprob(model, tokenizer, prompt: str, completion: str) -> CompletionScore:
    """Log-prob of `completion` given `prompt`, scored token-by-token.

    We tokenize prompt and prompt+completion separately and take the suffix as
    the scored region. A single leading space is folded into the completion so
    BPE boundary effects are consistent across options.
    """
    device = next(model.parameters()).device

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + completion, add_special_tokens=True).input_ids

    # Guard the prefix assumption: if tokenization of the prompt is not a strict
    # prefix of the joint tokenization, fall back to the longest common prefix.
    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:
        n_prompt = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            n_prompt += 1

    if len(full_ids) <= n_prompt:
        # Completion tokenized to nothing distinct; treat as a single eos-like step.
        return CompletionScore(sum_logp=0.0, n_tokens=0, mean_logp=0.0)

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits  # [1, T, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # token at position i is predicted by logits at position i-1
    total = 0.0
    n = 0
    for pos in range(n_prompt, len(full_ids)):
        tok_id = full_ids[pos]
        total += log_probs[0, pos - 1, tok_id].item()
        n += 1
    mean = total / n if n else 0.0
    return CompletionScore(sum_logp=total, n_tokens=n, mean_logp=mean)


@torch.no_grad()
def stance_score(model, tokenizer, prompt: str, option_a: str, option_b: str) -> dict:
    """Length-normalized stance score for one probe presentation.

    Returns the score plus the component pieces, so analysis can inspect whether
    a shift is driven by one option moving rather than a genuine A-vs-B swing.
    """
    a = completion_logprob(model, tokenizer, prompt, option_a)
    b = completion_logprob(model, tokenizer, prompt, option_b)
    return {
        "stance": a.mean_logp - b.mean_logp,
        "logp_a": a.mean_logp,
        "logp_b": b.mean_logp,
        "n_tok_a": a.n_tokens,
        "n_tok_b": b.n_tokens,
    }


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 60, do_sample: bool = False) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

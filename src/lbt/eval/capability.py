"""Capability & safety sanity checks per fine-tuned arm (SPEC §2.6).

- MMLU: stratified 5% sample, scored by per-choice logprob (deterministic).
  Flag if an arm drops > 2 points vs its own base model.
- Perplexity on a held-out neutral corpus split; flag if > 5% above base.
- 50-prompt refusal mini-battery (XSTest-style mix of safe-but-sensitive-sounding
  prompts and ones a tuned assistant should refuse). Exploratory link to the
  emergent-misalignment literature; we report drift, we don't gate on it.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch

from ..datagen.validators import looks_like_refusal
from ..modeling import LoadedModel, chat_prompt
from ..scoring import completion_logprobs_batched, generate

_MMLU_CHOICES = ("A", "B", "C", "D")


def mmlu_sample(frac: float, seed: int = 0) -> list[dict[str, Any]]:
    """Stratified-by-subject sample of the MMLU test split."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[int]] = {}
    for i, subj in enumerate(ds["subject"]):
        by_subject.setdefault(subj, []).append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for _subj, idxs in sorted(by_subject.items()):
        k = max(1, round(len(idxs) * frac))
        picked.extend(rng.sample(idxs, k))
    return [ds[i] for i in sorted(picked)]


def mmlu_accuracy(
    loaded: LoadedModel, questions: list[dict[str, Any]], batch_size: int = 16
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    for q in questions:
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip(_MMLU_CHOICES, q["choices"], strict=False)
        )
        prompt = chat_prompt(
            loaded.tokenizer,
            f"{q['question']}\n{choices}\nAnswer with the letter only.",
        )
        for label in _MMLU_CHOICES:
            pairs.append((prompt, label))
    scores = completion_logprobs_batched(loaded.model, loaded.tokenizer, pairs, batch_size)
    correct = 0
    for i, q in enumerate(questions):
        opts = [scores[4 * i + j].mean_logp for j in range(4)]
        if int(max(range(4), key=opts.__getitem__)) == int(q["answer"]):
            correct += 1
    return {"n": len(questions), "accuracy": correct / len(questions) if questions else None}


@torch.no_grad()
def perplexity(loaded: LoadedModel, texts: list[str], max_len: int = 1024) -> float:
    """Token-weighted perplexity over a held-out neutral text sample."""
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = loaded.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(loaded.device)
        if input_ids.shape[1] < 2:
            continue
        out = loaded.model(input_ids=input_ids, labels=input_ids)
        n = input_ids.shape[1] - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


def refusal_battery(
    loaded: LoadedModel, prompts: list[dict[str, Any]], max_new_tokens: int = 100
) -> dict[str, Any]:
    """Run the refusal mini-battery; report refusal rate by expected behavior."""
    rows = []
    for p in prompts:
        prompt = chat_prompt(loaded.tokenizer, p["prompt"])
        text = generate(loaded.model, loaded.tokenizer, prompt, max_new_tokens=max_new_tokens)
        rows.append(
            {
                "id": p["id"],
                "expect": p["expect"],
                "refused": looks_like_refusal(text),
                "response": text[:300],
            }
        )
    by_expect: dict[str, dict[str, float]] = {}
    for expect in ("comply", "refuse"):
        subset = [r for r in rows if r["expect"] == expect]
        if subset:
            by_expect[expect] = {
                "n": len(subset),
                "refusal_rate": sum(r["refused"] for r in subset) / len(subset),
            }
    return {"rows": rows, "summary": by_expect}

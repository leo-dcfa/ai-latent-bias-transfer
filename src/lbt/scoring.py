"""Log-prob scoring primitives — the measurement core of the experiment.

The primary endpoint (SPEC §2.6.1) is a contrast of completion log-probs:

    score = logP(pro_completion | prompt) - logP(anti_completion | prompt)

Length-normalized (per-token mean) so option pairs of unequal token length do
not confound stance with length. Pure forward passes: deterministic, cheap,
and identical machinery for base and fine-tuned arms.

Everything here is unit-tested with stub models (tests/test_scoring.py) because
a silent off-by-one in the scored region would corrupt every result downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompletionScore:
    sum_logp: float
    n_tokens: int

    @property
    def mean_logp(self) -> float:
        return self.sum_logp / self.n_tokens if self.n_tokens else 0.0


def _prefix_len(prompt_ids: list[int], full_ids: list[int]) -> int:
    """Length of the prompt's tokenization inside the joint tokenization.

    Tokenizers may merge across the prompt/completion boundary; fall back to the
    longest common prefix when the strict-prefix assumption fails.
    """
    n = len(prompt_ids)
    if full_ids[:n] == prompt_ids:
        return n
    n = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):
        if a != b:
            break
        n += 1
    return n


@torch.no_grad()
def completion_logprobs_batched(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int = 16,
) -> list[CompletionScore]:
    """Score logP(completion | prompt) for many (prompt, completion) pairs.

    Right-padded batches; the scored region is the completion suffix of each
    joint tokenization. Returns scores in input order.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    tokenized: list[tuple[list[int], int]] = []  # (full_ids, n_prompt)
    for prompt, completion in pairs:
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        full_ids = tokenizer(prompt + completion, add_special_tokens=True).input_ids
        tokenized.append((full_ids, _prefix_len(prompt_ids, full_ids)))

    out: list[CompletionScore] = []
    for start in range(0, len(tokenized), batch_size):
        chunk = tokenized[start : start + batch_size]
        max_len = max(len(ids) for ids, _ in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, (ids, _) in enumerate(chunk):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        for i, (ids, n_prompt) in enumerate(chunk):
            if len(ids) <= n_prompt:
                out.append(CompletionScore(sum_logp=0.0, n_tokens=0))
                continue
            # token at position p is predicted by logits at position p-1
            positions = torch.arange(n_prompt, len(ids))
            tok_ids = torch.tensor([ids[p] for p in positions])
            lp = log_probs[i, positions - 1, tok_ids]
            out.append(CompletionScore(sum_logp=float(lp.sum()), n_tokens=len(ids) - n_prompt))
    return out


@torch.no_grad()
def stance_scores_batched(
    model,
    tokenizer,
    items: list[tuple[str, str, str]],
    batch_size: int = 16,
) -> list[dict[str, float]]:
    """Length-normalized stance scores for (prompt, option_pro, option_anti) triples.

    Returns component pieces alongside the contrast so analysis can detect a
    shift driven by one option's fluency rather than a genuine pro-vs-anti swing.
    """
    pairs: list[tuple[str, str]] = []
    for prompt, pro, anti in items:
        pairs.append((prompt, pro))
        pairs.append((prompt, anti))
    scored = completion_logprobs_batched(model, tokenizer, pairs, batch_size=batch_size)
    out: list[dict[str, float]] = []
    for i in range(len(items)):
        a, b = scored[2 * i], scored[2 * i + 1]
        out.append(
            {
                "stance": a.mean_logp - b.mean_logp,
                "logp_pro": a.mean_logp,
                "logp_anti": b.mean_logp,
                "n_tok_pro": a.n_tokens,
                "n_tok_anti": b.n_tokens,
            }
        )
    return out


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    do_sample: bool = False,
) -> str:
    """Greedy (temp 0) generation for forced-choice, Likert, and open-ended formats."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, enc.input_ids.shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)

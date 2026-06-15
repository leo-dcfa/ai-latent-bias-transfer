"""Activation extraction at the elicitation position for a set of prompts (§3.1-3.2).

Shared by direction extraction (on source-domain framed texts) and the
representational-transfer metric (on held-out target prompts). Returns one
vector per prompt per layer, taken at the last content token (SPEC §3.1), with a
mean-pooled variant available as a robustness check.
"""

from __future__ import annotations

import numpy as np
import torch

from ..modeling import LoadedModel, chat_prompt
from .hooks import capture_resid, decoder_layers


def n_layers(model) -> int:
    return len(decoder_layers(model))


@torch.no_grad()
def activations_for_prompts(
    loaded: LoadedModel,
    prompts: list[str],
    layers: list[int] | None = None,
    position: str = "last_content_token",
    batch_size: int = 16,
) -> dict[int, np.ndarray]:
    """Capture residual-stream vectors for each prompt at each layer.

    `prompts` are already chat-formatted (use chat_prompt). position:
    "last_content_token" takes the final non-pad token; "mean_pooled" averages
    over the content tokens. Returns {layer: array[n_prompts, d_model]}.
    """
    model, tokenizer = loaded.model, loaded.tokenizer
    if layers is None:
        layers = list(range(n_layers(model)))

    collected: dict[int, list[np.ndarray]] = {ell: [] for ell in layers}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc.input_ids.to(loaded.device)
        attn = enc.attention_mask.to(loaded.device)
        with capture_resid(model, layers) as store:
            model(input_ids=input_ids, attention_mask=attn)
        for ell in layers:
            acts = store[ell]  # [B, T, D] cpu float32
            for b in range(acts.shape[0]):
                mask = attn[b].to("cpu").bool()
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if position == "mean_pooled":
                    vec = acts[b, idxs, :].mean(0)
                else:  # last content token
                    vec = acts[b, idxs[-1], :]
                collected[ell].append(vec.numpy())
    return {ell: np.stack(v) for ell, v in collected.items()}


def chat_format_all(loaded: LoadedModel, raw_prompts: list[str]) -> list[str]:
    return [chat_prompt(loaded.tokenizer, p) for p in raw_prompts]

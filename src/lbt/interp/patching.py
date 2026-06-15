"""Layer-localization via residual patching (SPEC §3.4).

For each target prompt, capture the framed model's residual stream at the
elicitation token positions, transplant it into BASE one layer at a time, and
measure how much of the framed model's stance score is recovered. Output: a
per-layer recovery curve (the localization heatmap row for that model family).
"""

from __future__ import annotations

import numpy as np
import torch

from ..modeling import LoadedModel, chat_prompt
from ..scoring import stance_scores_batched
from .activations import activations_for_prompts
from .hooks import patch_resid


def _item_triples(loaded: LoadedModel, items: list[dict]) -> list[tuple[str, str, str]]:
    """One (prompt, pro, anti) per item, using the first contrast template."""
    out = []
    for item in items:
        c = item["contrasts"][0]
        out.append((chat_prompt(loaded.tokenizer, c["question"]), c["pro"], c["anti"]))
    return out


def donor_activations(
    framed: LoadedModel, items: list[dict], layers: list[int]
) -> dict[int, np.ndarray]:
    """Framed model's last-token residual at each layer, one row per item."""
    prompts = [chat_prompt(framed.tokenizer, item["contrasts"][0]["question"]) for item in items]
    return activations_for_prompts(framed, prompts, layers=layers, position="last_content_token")


def patch_recovery(
    base: LoadedModel,
    items: list[dict],
    donor: dict[int, np.ndarray],
    layers: list[int],
    base_mean: float,
    framed_mean: float,
) -> dict[int, dict[str, float]]:
    """Per-layer stance recovery when BASE's last-token resid is patched from framed.

    recovery = (patched − base) / (framed − base): 1.0 means patching at this
    layer fully reproduces the framed model's stance; ~0 means it does nothing.
    """
    triples = _item_triples(base, items)
    out: dict[int, dict[str, float]] = {}
    denom = framed_mean - base_mean
    for layer in layers:
        stances: list[float] = []
        for i, (prompt, pro, anti) in enumerate(triples):
            d = torch.tensor(donor[layer][i], dtype=torch.float32)
            with patch_resid(base.model, layer, d, positions="last"):
                s = stance_scores_batched(base.model, base.tokenizer, [(prompt, pro, anti)])
            stances.append(s[0]["stance"])
        patched_mean = float(np.mean(stances))
        out[layer] = {
            "patched_mean": patched_mean,
            "recovery": float((patched_mean - base_mean) / denom) if abs(denom) > 1e-9 else float("nan"),
        }
    return out

"""Layer-localization via residual patching (SPEC §3.4).

For each target prompt, capture the framed model's residual stream at the
elicitation token, transplant it into BASE one layer at a time, and measure how
much of the framed model's stance is recovered. Output: a per-layer recovery
curve (the localization heatmap row for that model family).

Stance here is the decision-anchored letter contrast logP(' A') - logP(' B') on
the forced-choice prompt (oriented + = pro-change) — the same valid measure the
behavioral headline uses, not the bare-verdict-token contrast (which was found to
misread fine-tuned models; see reports/PHASE2_PLAIN_SUMMARY.md). Donor activations
and the patched forward pass therefore both use the forced-choice prompt so the
patched position lines up.
"""

from __future__ import annotations

import numpy as np
import torch

from ..modeling import LoadedModel, chat_prompt
from ..scoring import completion_logprobs_batched
from .activations import activations_for_prompts
from .hooks import patch_resid


def _fc_prompt(loaded: LoadedModel, item: dict) -> str:
    """Forced-choice prompt in canonical order (A = go ahead / pro-change)."""
    return chat_prompt(loaded.tokenizer, item["forced_choice"]["question"])


def _letter_stance(loaded: LoadedModel, prompt: str) -> float:
    """logP(' A') - logP(' B') on a forced-choice prompt where A = pro-change."""
    a, b = completion_logprobs_batched(loaded.model, loaded.tokenizer, [(prompt, " A"), (prompt, " B")])
    return a.sum_logp - b.sum_logp


def donor_activations(
    framed: LoadedModel, items: list[dict], layers: list[int]
) -> dict[int, np.ndarray]:
    """Framed model's last-token residual at each layer, one row per item."""
    prompts = [_fc_prompt(framed, item) for item in items]
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
    base_mean / framed_mean must be letter-stance means on the same forced-choice
    prompts (see steering / run_interp).
    """
    prompts = [_fc_prompt(base, item) for item in items]
    out: dict[int, dict[str, float]] = {}
    denom = framed_mean - base_mean
    for layer in layers:
        stances: list[float] = []
        for i, prompt in enumerate(prompts):
            d = torch.tensor(donor[layer][i], dtype=torch.float32)
            with patch_resid(base.model, layer, d, positions="last"):
                stances.append(_letter_stance(base, prompt))
        patched_mean = float(np.mean(stances))
        out[layer] = {
            "patched_mean": patched_mean,
            "recovery": float((patched_mean - base_mean) / denom) if abs(denom) > 1e-9 else float("nan"),
        }
    return out


def letter_stance_mean(loaded: LoadedModel, items: list[dict]) -> float:
    """Mean letter-stance over items' forced-choice prompts (for base/framed refs)."""
    return float(np.mean([_letter_stance(loaded, _fc_prompt(loaded, item)) for item in items]))

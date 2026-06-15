"""Representational-transfer metric (SPEC §3.2) and logit lens (§3.6).

Projection: for held-out target prompts, project activations onto the unit
stance direction at each layer, comparing BASE to each fine-tuned arm. H2
prediction: FRAME+ shifts negative, FRAME− positive, NEUTRAL ≈ 0 — on prompts
the training data never touched.
"""

from __future__ import annotations

import numpy as np


def project(acts: np.ndarray, unit_dir: np.ndarray) -> np.ndarray:
    """Per-prompt scalar projection onto a unit direction."""
    return acts @ unit_dir


def projection_shift(
    base_acts: dict[int, np.ndarray],
    arm_acts: dict[int, np.ndarray],
    directions: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Mean change in stance-direction projection (arm − base) per layer.

    Same prompts in both, so the per-prompt difference is paired; we report the
    mean paired shift and its standard error.
    """
    out: dict[int, dict[str, float]] = {}
    for layer in sorted(directions):
        u = directions[layer]
        base_p = project(base_acts[layer], u)
        arm_p = project(arm_acts[layer], u)
        diff = arm_p - base_p
        out[layer] = {
            "mean_shift": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
            "base_mean_proj": float(base_p.mean()),
            "arm_mean_proj": float(arm_p.mean()),
            "n": int(len(diff)),
        }
    return out

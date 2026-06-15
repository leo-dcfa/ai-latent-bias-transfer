"""LoRA delta analysis (SPEC §3.5).

Per adapted module, ΔW = (α/r)·B·A. We report:
- effective rank (singular-value concentration / participation ratio of ΔW), and
- an amplification ratio: how much ΔW amplifies inputs along the stance direction
  vs along random directions, mapped into each module's input space.

The spec flags raw weight geometry as coordinate-fragile and prefers an
activation-delta formulation; both are provided. The weight-based amplification
needs d̂ expressed in the module's *input* space, which is only well-defined for
modules whose input is the residual stream (q/k/v/gate/up projections). For o/down
projections (input = attention/MLP intermediate) we report effective rank only.
"""

from __future__ import annotations

import numpy as np


def delta_w(lora_a: np.ndarray, lora_b: np.ndarray, alpha: int, r: int) -> np.ndarray:
    """ΔW = (α/r)·B·A. lora_a: [r, in], lora_b: [out, r] -> ΔW: [out, in]."""
    return (alpha / r) * (lora_b @ lora_a)


def effective_rank(matrix: np.ndarray) -> dict[str, float]:
    """Participation ratio of singular values (a continuous 'effective rank') and
    the top-singular-value energy fraction."""
    sv = np.linalg.svd(matrix, compute_uv=False)
    sv2 = sv**2
    total = sv2.sum()
    if total <= 0:
        return {"effective_rank": 0.0, "top1_energy": 0.0, "n_sv": int(len(sv))}
    p = sv2 / total
    eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))  # exp of spectral entropy
    return {"effective_rank": eff, "top1_energy": float(p[0]), "n_sv": int(len(sv))}


def amplification_ratio(
    delta: np.ndarray, dir_in: np.ndarray, n_random: int = 16, seed: int = 0
) -> float:
    """‖ΔW·d̂_in‖ vs the mean ‖ΔW·r̂‖ over random unit r̂ in the input space (§3.5).

    >1 means the LoRA update amplifies the stance direction more than chance.
    `dir_in` must be a unit vector in ΔW's input space (length = ΔW.shape[1]).
    """
    d = dir_in / (np.linalg.norm(dir_in) + 1e-12)
    along = float(np.linalg.norm(delta @ d))
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        v = rng.standard_normal(delta.shape[1])
        v /= np.linalg.norm(v)
        rand.append(float(np.linalg.norm(delta @ v)))
    base = float(np.mean(rand)) if rand else float("nan")
    return along / base if base > 0 else float("nan")


def activation_delta_cosine(
    base_acts: np.ndarray, framed_acts: np.ndarray, unit_dir: np.ndarray
) -> dict[str, float]:
    """Spec's preferred, coordinate-robust formulation: cosine of the mean
    fine-tuned-minus-base activation delta with the stance direction (§3.5)."""
    delta = framed_acts.mean(0) - base_acts.mean(0)
    norm = np.linalg.norm(delta)
    cos = float(delta @ unit_dir / norm) if norm > 0 else 0.0
    return {"delta_norm": float(norm), "cosine_with_stance": cos}

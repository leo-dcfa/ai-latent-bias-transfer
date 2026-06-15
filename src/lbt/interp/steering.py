"""Causal steering and directional ablation (SPEC §3.3).

Steering: add α·d̂ to BASE's residual stream at ℓ* during the logprob battery;
sweep α as fractions of the typical residual norm at ℓ*. Report the dose-response
curve of behavioral shift vs fluency cost (perplexity on neutral text).

Ablation: project d̂ out of a framed model's residual stream during the battery;
report the fraction of the FRAME-vs-NEUTRAL gap removed.

Controls (mandatory, §3.3): random unit directions with matched norm must do
neither — provided here so callers always run them alongside.
"""

from __future__ import annotations

import numpy as np
import torch

from ..eval.battery import logprob_rows
from ..modeling import LoadedModel
from .hooks import ablate_direction, add_direction


def typical_resid_norm(acts: np.ndarray) -> float:
    """Median L2 norm of residual vectors at a layer — the scale steering α
    fractions are measured against."""
    return float(np.median(np.linalg.norm(acts, axis=1)))


def random_unit_like(dir_unit: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dir_unit.shape[0])
    return v / np.linalg.norm(v)


def mean_stance(rows: list[dict]) -> float:
    vals = [r["stance"] for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def steer_battery(
    loaded: LoadedModel,
    items: list[dict],
    unit_dir: np.ndarray,
    alpha: float,
    layer: int,
    positions: str = "all",
    batch_size: int = 16,
) -> list[dict]:
    """Run the logprob battery on BASE with +α·d̂ added at `layer`."""
    vec = torch.tensor(unit_dir * alpha, dtype=torch.float32)
    with add_direction(loaded.model, layer, vec, positions=positions):
        rows = logprob_rows(loaded, items, batch_size=batch_size)
    for r in rows:
        r["alpha"] = alpha
        r["layer"] = layer
    return rows


def ablate_battery(
    loaded: LoadedModel,
    items: list[dict],
    unit_dir: np.ndarray,
    layer: int,
    positions: str = "all",
    batch_size: int = 16,
) -> list[dict]:
    """Run the logprob battery with d̂ projected out at `layer`."""
    u = torch.tensor(unit_dir, dtype=torch.float32)
    with ablate_direction(loaded.model, layer, u, positions=positions):
        rows = logprob_rows(loaded, items, batch_size=batch_size)
    for r in rows:
        r["layer"] = layer
        r["ablated"] = True
    return rows


def gap_removed_fraction(framed_mean: float, neutral_mean: float, ablated_mean: float) -> float:
    """Fraction of the FRAME-vs-NEUTRAL gap that ablation removes (§3.3).

    1.0 = ablation moves the framed model all the way to NEUTRAL; 0.0 = no effect.
    """
    gap = framed_mean - neutral_mean
    if abs(gap) < 1e-9:
        return float("nan")
    return float((framed_mean - ablated_mean) / gap)

"""Stretch (SPEC §3.7): developmental dynamics via Local Learning Coefficient.

Two analyses over the fractional checkpoints (§2.5):
(a) re-run the H2 projection metric at each checkpoint to locate *when*
    representational transfer emerges (pure reuse of projections.py — no extra
    dependency), and
(b) estimate LLC trajectories with the `devinterp` library (SGLD defaults,
    restricted to LoRA parameters). This is explicitly exploratory.

`devinterp` is an optional dependency (pip install '.[devinterp]'); the import is
deferred so Phases 0–5 never require it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def estimate_llc(
    model,
    data_loader,
    loss_fn: Callable,
    *,
    num_chains: int = 4,
    num_draws: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """LLC estimate via devinterp SGLD (exploratory; document caveats in the report).

    Restricts sampling to LoRA parameters by freezing everything else before the
    call. Returns the devinterp results dict (mean LLC + per-chain trace).
    """
    try:
        from devinterp.optim.sgld import SGLD
        from devinterp.slt import estimate_learning_coeff_with_summary
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "devinterp not installed. Install with: uv sync --extra devinterp"
        ) from e

    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)

    return estimate_learning_coeff_with_summary(
        model,
        loader=data_loader,
        evaluate=loss_fn,
        sampling_method=SGLD,
        num_chains=num_chains,
        num_draws=num_draws,
        seed=seed,
    )

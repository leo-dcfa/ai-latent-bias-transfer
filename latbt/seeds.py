"""Deterministic seeding across python/numpy/torch.

Every script calls set_all_seeds(cfg.seed) at startup and logs it, so the two
adapters per model and the eval pass are reproducible. The seed is part of what
is held CONSTANT across models in the comparison design.
"""

from __future__ import annotations

import os
import random


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Prefer determinism; bf16 matmul still has minor nondeterminism on GPU,
        # which is acceptable and recorded as a known limitation in the README.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

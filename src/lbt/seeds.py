"""Deterministic seeding across python/numpy/torch/cuda (SPEC §2.5).

Every entry point calls set_all_seeds() at startup and the seed is recorded in
the run metadata. bf16 GPU matmul retains minor nondeterminism; this is a known,
documented limitation rather than something we paper over.
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
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

"""Stance-direction extraction and probing on the BASE model (SPEC §3.1).

The stance direction at layer ℓ is the difference of class means of residual
activations between FRAME+ and FRAME− *source-domain* texts. A logistic probe
validates each layer; the analysis layer ℓ* is chosen by held-out probe accuracy.

Sign convention: the unit direction points from FRAME+ (precaution) toward
FRAME− (proaction), i.e. toward pro-change, matching the behavioral sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerDirection:
    layer: int
    direction: np.ndarray  # unit vector, frame_plus -> frame_minus
    raw_norm: float  # ‖mean(frame_minus) - mean(frame_plus)‖
    probe_acc: float  # held-out accuracy of a logistic probe at this layer


def diff_of_means(frame_plus: np.ndarray, frame_minus: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit direction pointing from FRAME+ mean to FRAME− mean, and its raw norm."""
    raw = frame_minus.mean(0) - frame_plus.mean(0)
    norm = float(np.linalg.norm(raw))
    unit = raw / norm if norm > 0 else raw
    return unit, norm


def probe_accuracy(
    frame_plus: np.ndarray, frame_minus: np.ndarray, val_frac: float = 0.25, seed: int = 0
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    x = np.concatenate([frame_plus, frame_minus])
    y = np.array([0] * len(frame_plus) + [1] * len(frame_minus))
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=val_frac, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))


def extract_directions(
    acts_plus: dict[int, np.ndarray],
    acts_minus: dict[int, np.ndarray],
    val_frac: float = 0.25,
    seed: int = 0,
) -> dict[int, LayerDirection]:
    """Per-layer stance direction + probe accuracy from matched activation sets."""
    out: dict[int, LayerDirection] = {}
    for layer in sorted(acts_plus):
        unit, norm = diff_of_means(acts_plus[layer], acts_minus[layer])
        acc = probe_accuracy(acts_plus[layer], acts_minus[layer], val_frac, seed)
        out[layer] = LayerDirection(layer=layer, direction=unit, raw_norm=norm, probe_acc=acc)
    return out


def select_layer(directions: dict[int, LayerDirection]) -> int:
    """ℓ* = the layer with the highest held-out probe accuracy (ties: earliest)."""
    return max(sorted(directions), key=lambda ell: directions[ell].probe_acc)

"""Figure generation from cached artifacts (SPEC §2.7, §4).

Every figure regenerates from cached eval/interp outputs with no GPU — that is
the Phase 3/4 acceptance criterion. Functions take already-loaded dicts/arrays
and an output path; the calling script handles file IO so figures are trivially
reproducible and testable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def behavioral_effect_figure(report: dict, out_path: str | Path) -> Path:
    """Forest-style plot of FRAME± d vs NEUTRAL with CIs, per family + combined."""
    families = list(report["families"])
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4), squeeze=False)
    for ax, family in zip(axes[0], families, strict=False):
        fam = report["families"][family]
        labels, points, lo, hi = [], [], [], []
        for arm in ("frame_plus", "frame_minus", "combined_directional"):
            entry = fam[arm]
            labels.append(arm.replace("frame_", "").replace("_", " "))
            points.append(entry["d"])
            ci = entry["ci"]
            lo.append(entry["d"] - ci[0])
            hi.append(ci[1] - entry["d"])
        y = range(len(labels))
        ax.errorbar(points, y, xerr=[lo, hi], fmt="o", capsize=4)
        ax.axvline(0, color="gray", lw=0.8)
        sesoi = report["sesoi_d"]
        ax.axvspan(-sesoi, sesoi, color="orange", alpha=0.1, label=f"SESOI ±{sesoi}")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlabel("effect size d (NEUTRAL-SD units)")
        ax.set_title(family)
        ax.legend(fontsize=8)
    fig.suptitle("Behavioral stance transfer to held-out targets")
    return _save(fig, out_path)


def probe_accuracy_figure(probe_acc: dict[int, float], selected_layer: int, out_path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = sorted(probe_acc)
    ax.plot(layers, [probe_acc[ell] for ell in layers], marker="o")
    ax.axvline(selected_layer, color="red", ls="--", label=f"ℓ* = {selected_layer}")
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_title("Stance-direction linear probe accuracy by layer")
    ax.legend()
    return _save(fig, out_path)


def projection_shift_figure(shifts_by_arm: dict[str, dict[int, dict]], out_path) -> Path:
    """Per-layer mean projection shift (arm − base) onto the stance direction."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, per_layer in shifts_by_arm.items():
        layers = sorted(per_layer)
        means = [per_layer[ell]["mean_shift"] for ell in layers]
        ses = [per_layer[ell]["se"] for ell in layers]
        ax.errorbar(layers, means, yerr=ses, marker="o", capsize=2, label=arm)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("projection shift (arm − base)")
    ax.set_title("Representational transfer on held-out targets (H2)")
    ax.legend()
    return _save(fig, out_path)


def steering_dose_response_figure(curve: list[dict], out_path) -> Path:
    """Behavioral stance shift vs fluency cost across steering α."""
    fig, ax = plt.subplots(figsize=(7, 4))
    alphas = [c["alpha"] for c in curve]
    ax.plot(alphas, [c["mean_stance"] for c in curve], marker="o", label="stance")
    ax.set_xlabel("steering α (fraction of typical resid norm)")
    ax.set_ylabel("mean stance score")
    ax2 = ax.twinx()
    ax2.plot(alphas, [c["perplexity"] for c in curve], marker="s", color="red", label="ppl")
    ax2.set_ylabel("perplexity (neutral text)")
    ax.set_title("Steering dose-response (§3.3)")
    return _save(fig, out_path)


def patching_heatmap_figure(recovery_by_family: dict[str, dict[int, dict]], out_path) -> Path:
    import numpy as np

    families = list(recovery_by_family)
    all_layers = sorted({ell for fam in recovery_by_family.values() for ell in fam})
    mat = np.array(
        [[recovery_by_family[f].get(ell, {}).get("recovery", np.nan) for ell in all_layers] for f in families]
    )
    fig, ax = plt.subplots(figsize=(max(6, len(all_layers) * 0.3), 1.5 + len(families)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(all_layers)))
    ax.set_xticklabels(all_layers, fontsize=7)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)
    ax.set_xlabel("patched layer")
    ax.set_title("Stance-score recovery by patched layer (§3.4)")
    fig.colorbar(im, ax=ax, label="recovery fraction")
    return _save(fig, out_path)

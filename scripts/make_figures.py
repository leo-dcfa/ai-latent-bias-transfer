"""Regenerate every report figure from cached artifacts (no GPU).

All multi-panel figures are stacked vertically (one model per row) to avoid the
label overlap of side-by-side layouts. Produces:
  - behavioral_effect_<metric>.png   (per-metric transfer, models stacked)
  - model_summary_<family>.png        (representational top + causal steering bottom)
  - patching_heatmap.png              (layer localization)
"""

from __future__ import annotations

import json

from _common import REPO_ROOT, base_parser
from lbt.config import load_config
from lbt.stats.figures import (
    asymmetry_figure,
    behavioral_effect_figure,
    model_summary_figure,
    patching_heatmap_figure,
)

FIG = REPO_ROOT / "reports" / "figures"


def _load(p):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    cfg = load_config(base_parser("Regenerate report figures.").parse_args().config)
    run = cfg.runs_dir / cfg.name
    made = []

    stats = _load(run / "stats" / "primary_report.json")
    if stats:
        for metric, m in stats["metrics"].items():
            t = m.get("transfer", {})
            if "families" in t:
                made.append(behavioral_effect_figure({**t, "sesoi_d": stats["sesoi_d"]},
                                                      FIG / f"behavioral_effect_{metric}.png"))

    # asymmetry: trained-domain forced-choice means + held-out per-arm d (letter-logprob)
    diag = _load(run / "manipulation_metrics_diagnostic.json")
    if stats and diag:
        # the metric dict carries a "description" sibling — keep only the family entries
        src = {k: v for k, v in diag["metrics"]["forced_choice"].items() if isinstance(v, dict)}
        tll = stats["metrics"]["letter_logprob"]["transfer"].get("families", {})
        tgt = {fam: {"frame_plus": f["frame_plus"]["d"], "frame_minus": f["frame_minus"]["d"]}
               for fam, f in tll.items()}
        if tgt:
            made.append(asymmetry_figure(src, tgt, FIG / "asymmetry.png"))

    families = sorted({m.family for m in cfg.models})
    recov = {}
    for fam in families:
        proj = _load(run / "interp" / fam / "projection_shift.json")
        steer = _load(run / "interp" / fam / "steering.json")
        direc = _load(run / "interp" / fam / "direction.json")
        pat = _load(run / "interp" / fam / "patching.json")
        if proj and steer and direc:
            made.append(model_summary_figure(fam, proj, steer, direc["l_star"],
                                             FIG / f"model_summary_{fam}.png"))
        if pat and pat.get("recovery"):
            recov[fam] = pat["recovery"]
    if recov:
        made.append(patching_heatmap_figure(recov, FIG / "patching_heatmap.png"))

    for p in made:
        print("wrote", p)
    print(f"\n{len(made)} figures -> {FIG}")


if __name__ == "__main__":
    main()

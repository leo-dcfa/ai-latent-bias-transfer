"""Phase 3: statistics + behavioral figures from cached eval rows (SPEC §2.7).

Reads runs/<exp>/eval/*/target_logprob.jsonl, runs the primary endpoint analysis
(per-family d with hierarchical-bootstrap CIs, TOST equivalence, Holm-corrected
directional test), writes a JSON report + the behavioral effect figure. No GPU.

Also reports the manipulation check (source-domain d >= 0.5 gate, §2.6) so a
transfer null is never interpreted without confirming the framing was instilled.
"""

from __future__ import annotations

import json

from _common import REPO_ROOT, base_parser
from lbt.config import load_config
from lbt.io import load_jsonl
from lbt.stats.analysis import aggregate_logprob_to_item, analyze_primary, cohens_d_vs_neutral
from lbt.stats.figures import behavioral_effect_figure
from lbt.train.sft import cell_name


def _collect_item_rows(cfg, family_models, split):
    """Gather per-item logprob stance rows for all arms of given model keys."""
    rows = []
    for model_spec in family_models:
        for condition in cfg.conditions:
            for seed in cfg.seeds:
                cell = cell_name(model_spec.key, condition, seed)
                path = cfg.runs_dir / cfg.name / "eval" / cell / f"{split}_logprob.jsonl"
                if path.exists():
                    rows.extend(load_jsonl(path))
    agg = aggregate_logprob_to_item(rows)
    return list(agg.values())


def main() -> None:
    parser = base_parser("Behavioral statistics + figures from cached rows.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scfg = cfg.stats

    families = sorted({m.family for m in cfg.models})
    target_by_family, source_by_family = {}, {}
    for fam in families:
        fam_models = [m for m in cfg.models if m.family == fam]
        target_by_family[fam] = _collect_item_rows(cfg, fam_models, "target")
        source_by_family[fam] = _collect_item_rows(cfg, fam_models, "source")

    report = analyze_primary(
        target_by_family,
        n_resamples=int(scfg["bootstrap_resamples"]),
        sesoi_d=float(scfg["sesoi_d"]),
        alpha=float(scfg["alpha"]),
    )

    # Manipulation check: source-domain d vs NEUTRAL must reach 0.5 for a framed arm.
    manip = {}
    for fam, rows in source_by_family.items():
        if not rows:
            continue
        manip[fam] = {
            arm: cohens_d_vs_neutral(rows, arm) for arm in ("frame_plus", "frame_minus")
        }
    report["manipulation_check"] = manip
    report["manipulation_gate_d"] = 0.5

    out_dir = cfg.runs_dir / cfg.name / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primary_report.json").write_text(json.dumps(report, indent=2, default=str))

    fig_dir = REPO_ROOT / "reports" / "figures"
    if any(target_by_family.values()):
        behavioral_effect_figure(report, fig_dir / "behavioral_effect.png")

    print(json.dumps(report["families"], indent=2, default=str))
    print("\nmanipulation check (source d vs NEUTRAL, gate=0.5):")
    print(json.dumps(manip, indent=2, default=str))
    print(f"\nreport -> {out_dir / 'primary_report.json'}")
    print(f"figure -> {fig_dir / 'behavioral_effect.png'}")


if __name__ == "__main__":
    main()

"""Phase 3: statistics + figures from cached eval rows, under EVERY stance metric.

Per the locked preregistration, no single metric is the hand-picked headline. We
report the analysis under each measure side by side: bare-token logprob,
letter-logprob (decision-anchored), forced-choice, and Likert. A "transfer"
conclusion requires the behavioral measures to agree; disagreement is itself
reported. (Background: reports/PHASE2_PLAIN_SUMMARY.md.)

For each metric we emit, per model family:
  - manipulation check: source-domain d vs NEUTRAL per framed arm (gate |d|>=0.5),
  - transfer test: the primary endpoint on held-out TARGET items (per-arm d with
    hierarchical-bootstrap CIs, TOST equivalence, Holm-corrected directional test).

No GPU; regenerates from cached rows with one command.
"""

from __future__ import annotations

import json

from _common import REPO_ROOT, base_parser
from lbt.config import load_config
from lbt.io import load_jsonl
from lbt.stats.analysis import aggregate_logprob_to_item, analyze_primary, cohens_d_vs_neutral
from lbt.stats.figures import behavioral_effect_figure
from lbt.train.sft import cell_name

METRICS = ("logprob", "letter_logprob", "forced_choice", "likert")


def _collect_item_rows(cfg, family_models, split, metric):
    """Per-item stance rows for all arms of given models, for one split × metric.

    Rows with an unparsed stance (None — e.g. a Likert answer that wasn't a number)
    are dropped before aggregation.
    """
    rows = []
    for model_spec in family_models:
        for condition in cfg.conditions:
            for seed in cfg.seeds:
                cell = cell_name(model_spec.key, condition, seed)
                path = cfg.runs_dir / cfg.name / "eval" / cell / f"{split}_{metric}.jsonl"
                if path.exists():
                    rows.extend(r for r in load_jsonl(path) if r.get("stance") is not None)
    return list(aggregate_logprob_to_item(rows).values())


def main() -> None:
    parser = base_parser("Behavioral statistics under every metric, from cached rows.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scfg = cfg.stats
    families = sorted({m.family for m in cfg.models})

    report: dict = {"metrics": {}, "manipulation_gate_d": 0.5,
                    "sesoi_d": float(scfg["sesoi_d"]), "alpha": float(scfg["alpha"])}
    fig_dir = REPO_ROOT / "reports" / "figures"

    for metric in METRICS:
        target_by_family, source_by_family = {}, {}
        for fam in families:
            fam_models = [m for m in cfg.models if m.family == fam]
            target_by_family[fam] = _collect_item_rows(cfg, fam_models, "target", metric)
            source_by_family[fam] = _collect_item_rows(cfg, fam_models, "source", metric)

        has_target = any(target_by_family.values())
        primary = (
            analyze_primary(
                target_by_family,
                n_resamples=int(scfg["bootstrap_resamples"]),
                sesoi_d=float(scfg["sesoi_d"]),
                alpha=float(scfg["alpha"]),
            )
            if has_target
            else {"note": "no target rows cached yet (transfer test not run / pre-lock)"}
        )
        manip = {
            fam: {arm: cohens_d_vs_neutral(rows, arm) for arm in ("frame_plus", "frame_minus")}
            for fam, rows in source_by_family.items()
            if rows
        }
        report["metrics"][metric] = {"transfer": primary, "manipulation_check": manip}

        if has_target:
            behavioral_effect_figure(primary, fig_dir / f"behavioral_effect_{metric}.png")

    out_dir = cfg.runs_dir / cfg.name / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primary_report.json").write_text(json.dumps(report, indent=2, default=str))

    # console summary
    for metric in METRICS:
        m = report["metrics"][metric]
        print(f"\n===== {metric} =====")
        print("manipulation (source d vs NEUTRAL, gate 0.5):")
        for fam, arms in m["manipulation_check"].items():
            print(f"  {fam}: " + ", ".join(f"{a}={d:+.3f}" for a, d in arms.items()))
        if "families" in m["transfer"]:
            print("transfer (target combined directional d, predict >0):")
            for fam, f in m["transfer"]["families"].items():
                cd = f["combined_directional"]
                print(f"  {fam}: d={cd['d']:+.3f}  CI[{cd['ci'][0]:+.3f},{cd['ci'][1]:+.3f}]")
        else:
            print("transfer: " + m["transfer"].get("note", ""))
    print(f"\nreport -> {out_dir / 'primary_report.json'}")


if __name__ == "__main__":
    main()

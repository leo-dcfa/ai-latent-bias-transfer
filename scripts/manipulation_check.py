"""Phase 2 gate: the manipulation check (SPEC §2.6) — SOURCE domains only.

Runs the primary (logprob) stance battery on the *source* domains for BASE + every
trained arm, and reports each framed arm's source-domain effect size vs NEUTRAL.
An arm qualifies for the transfer test only if its source shift is d >= 0.5 in the
expected direction (FRAME+ negative / precaution, FRAME- positive / proaction).

Deliberately does NOT touch the held-out TARGET items: the transfer endpoint must
not be scored until the preregistration is locked (SPEC §2.7 / Phase 3). This script
is safe to run pre-lock — source items are a separate, pre-registered gate.

Source rows are cached to runs/<exp>/eval/<cell>/source_logprob.jsonl, which the
later Phase 3 eval reuses, so nothing is wasted.
"""

from __future__ import annotations

import json

from _common import base_parser
from lbt.config import load_config
from lbt.eval.battery import logprob_rows
from lbt.io import load_jsonl, write_jsonl
from lbt.modeling import free, load_for_eval
from lbt.seeds import set_all_seeds
from lbt.stats.analysis import aggregate_logprob_to_item, cohens_d_vs_neutral
from lbt.train.sft import cell_name, find_adapter

# Expected direction of the source-domain shift if the framing instilled.
EXPECTED_SIGN = {"frame_plus": -1.0, "frame_minus": +1.0}
GATE_D = 0.5


def _eval_dir(cfg, cell):
    d = cfg.runs_dir / cfg.name / "eval" / cell
    d.mkdir(parents=True, exist_ok=True)
    return d


def _score_and_cache(cfg, loaded, cell, condition, seed, source_items, bs):
    rows = logprob_rows(loaded, source_items, batch_size=bs)
    for r in rows:
        r["seed"] = seed
        r["condition"] = condition
    write_jsonl(_eval_dir(cfg, cell) / "source_logprob.jsonl", rows)
    return rows


def main() -> None:
    parser = base_parser("Phase 2 manipulation check (source-domain battery).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bs = int(cfg.eval["batch_size"])
    source_items = load_jsonl(cfg.eval_dir / "source_items.jsonl")

    report = {"gate_d": GATE_D, "families": {}}
    for model_spec in cfg.models:
        family = model_spec.family
        rows: list[dict] = []

        set_all_seeds(0)
        base = load_for_eval(model_spec.model_id, None, label="base")
        rows += _score_and_cache(cfg, base, cell_name(model_spec.key, "base", 0), "base", 0, source_items, bs)
        free(base)

        for condition in cfg.conditions:
            for seed in cfg.seeds:
                cell = cell_name(model_spec.key, condition, seed)
                try:
                    adapter = find_adapter(cfg, model_spec.key, condition, seed)
                except FileNotFoundError:
                    print(f"skip (no adapter): {cell}")
                    continue
                set_all_seeds(seed)
                loaded = load_for_eval(model_spec.model_id, str(adapter), label=condition)
                rows += _score_and_cache(cfg, loaded, cell, condition, seed, source_items, bs)
                free(loaded)
                print(f"scored {cell}")

        # per-arm source-domain d vs NEUTRAL (item-level), pooled over seeds
        item_rows = list(aggregate_logprob_to_item(rows).values())
        fam = {}
        for arm in ("frame_plus", "frame_minus"):
            d = cohens_d_vs_neutral(item_rows, arm)
            expected = EXPECTED_SIGN[arm]
            qualifies = (d * expected) >= GATE_D
            fam[arm] = {
                "d_vs_neutral": d,
                "expected_sign": expected,
                "qualifies": bool(qualifies),
            }
        report["families"][family] = fam

    out = cfg.runs_dir / cfg.name / "manipulation_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print("\n=== MANIPULATION CHECK (source-domain d vs NEUTRAL, gate |d| >= 0.5 in expected dir) ===")
    any_qual = False
    for family, fam in report["families"].items():
        for arm, r in fam.items():
            mark = "QUALIFIES" if r["qualifies"] else "below gate"
            any_qual = any_qual or r["qualifies"]
            print(f"  {family:6s} {arm:12s} d={r['d_vs_neutral']:+.3f} (expect sign {r['expected_sign']:+.0f})  {mark}")
    print(f"\nreport -> {out}")
    if not any_qual:
        print(
            "\nNO ARM QUALIFIES. Per SPEC §2.6 a transfer null here is uninterpretable — "
            "invoke the pre-registered escalation ladder (§2.4) before any conclusion."
        )


if __name__ == "__main__":
    main()

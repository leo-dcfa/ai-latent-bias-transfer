"""Phase 1 discipline: pilot a SMALL corpus per arm and run the validators before
committing to the full (~21 h) generation (SPEC §5 — fix templates at small scale).

Generates `--n` docs/arm into data/corpora/pilot/, then runs every §2.4 gate on the
pilot. The framing-classifier AUC and surface-stat matching are meaningful even at
n~200, so this catches diversity/length regressions in ~15 minutes instead of a day.

Usage:
    LBT_GEN_MODEL=gemma3:27b uv run python scripts/pilot_data.py --config configs/lbt2.yaml --n 200
"""

from __future__ import annotations

import json
import sys

from _common import base_parser
from lbt.config import load_blocklist, load_config
from lbt.datagen.generator import generate_corpus
from lbt.datagen.templates import n_unique_cells, template_version
from lbt.datagen.validators import validate_corpora
from lbt.io import load_jsonl
from lbt.seeds import set_all_seeds


def main() -> None:
    parser = base_parser("Pilot a small corpus per arm and validate before the full run.")
    parser.add_argument("--n", type=int, default=200, help="docs per arm for the pilot")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_all_seeds(0)

    pilot_dir = cfg.corpora_dir / "pilot"
    blocklist = load_blocklist(cfg.path(cfg.data["blocklist"]))
    terms = [t for v in blocklist.values() for t in v]

    print(f"templates {template_version()}  |  {n_unique_cells()} unique cells")
    print(f"piloting {args.n} docs/arm -> {pilot_dir}\n")

    corpora = {}
    for arm in cfg.conditions:
        meta = generate_corpus(
            arm=arm, n_target=args.n, datagen_cfg=cfg.datagen, data_cfg=cfg.data,
            blocklist_terms=terms, out_dir=pilot_dir,
        )
        print(f"  {arm}: yield={meta['yield_frac']:.0%} kept={meta['n_kept']} rejects={meta['reject_reasons']}")
        corpora[arm] = load_jsonl(pilot_dir / f"{arm}.jsonl")

    target_path = cfg.eval_dir / "target_items.jsonl"
    eval_texts = [it["scenario"] for it in load_jsonl(target_path)] if target_path.exists() else None

    print("\nrunning validators on pilot...")
    report = validate_corpora(corpora, terms, cfg.data, eval_texts=eval_texts)
    (pilot_dir / "validation_report.json").write_text(json.dumps(report, indent=2))

    def st(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    surf, fc, dd, ea = (
        report["surface"], report["framing_classifier"], report["dedup"], report["embedding_audit"]
    )
    lengths = {k: round(v) for k, v in surf["length_means"].items()}
    print(f"\nleakage            {st(report['leakage_passed'])}")
    print(f"surface stats      {st(surf['passed'])}  lengths={lengths}")
    print(f"refusals           {st(report['refusals_passed'])}")
    print(f"framing classifier {st(fc['passed'])}  AUC={fc['auc']:.3f}  NEUTRAL={fc['neutral_mean']:.3f}")
    print(f"dedup              {st(dd['passed'])}  near_pairs={len(dd['near_pairs'])}")
    print(f"embedding audit    {st(ea['passed'])}  max_sim={ea.get('max_sim_overall', float('nan')):.3f}")
    print(f"\nPILOT OVERALL: {st(report['passed'])}")
    if report["passed"]:
        print("→ gates pass at pilot scale. Safe to run the full corpus:  make gen-data && make validate")
    else:
        print("→ fix templates/config and re-pilot BEFORE the full run (SPEC §5).")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

"""Phase 1 gate: run every §2.4 validator on the generated corpora.

Blocking: exits nonzero if any gate fails. Writes a full report to
data/corpora/validation_report.json.

The embedding audit (§2.3.2) runs against the *held-out target* eval items only.
Source-domain eval items deliberately share domains with the training corpora
(they drive the §2.6 manipulation check), so they are *expected* to embed near
training docs — auditing them would flag by design. Only target items must stay
< the audit threshold from anything in training.
"""

from __future__ import annotations

import json
import sys

from _common import base_parser
from lbt.config import load_blocklist, load_config
from lbt.datagen.validators import validate_corpora
from lbt.io import load_jsonl


def main() -> None:
    parser = base_parser("Validate training corpora against all §2.4 gates.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    corpora = {}
    for arm in cfg.conditions:
        path = cfg.corpora_dir / f"{arm}.jsonl"
        if not path.exists():
            sys.exit(f"missing corpus {path}; run gen_data.py first")
        corpora[arm] = load_jsonl(path)

    blocklist = load_blocklist(cfg.path(cfg.data["blocklist"]))
    terms = [t for terms in blocklist.values() for t in terms]

    # Held-out TARGET items only — source items are meant to resemble training.
    eval_texts = []
    target_path = cfg.eval_dir / "target_items.jsonl"
    if target_path.exists():
        eval_texts = [item["scenario"] for item in load_jsonl(target_path)]

    print("running validators (embedding model load + classifier may take a minute)...")
    report = validate_corpora(corpora, terms, cfg.data, eval_texts=eval_texts or None)

    out = cfg.corpora_dir / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"\nleakage           {status(report['leakage_passed'])}")
    surf = report["surface"]
    print(
        f"surface stats     {status(surf['passed'])}  "
        f"(ks_ok={surf['ks_ok']}, length_mean_ok={surf['length_mean_ok']}, ttr_ok={surf['ttr_ok']})"
    )
    print(f"refusals          {status(report['refusals_passed'])}")
    fc = report["framing_classifier"]
    print(f"framing classifier{status(fc['passed'])}  AUC={fc['auc']:.3f}  NEUTRAL mean={fc['neutral_mean']:.3f}")
    print(f"dedup             {status(report['dedup']['passed'])}")
    print(f"embedding audit   {status(report['embedding_audit']['passed'])}")
    print(f"\noverall: {status(report['passed'])}   report -> {out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

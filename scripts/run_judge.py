"""Phase 3 (secondary): judge the cached open-ended responses (SPEC §2.6.4).

Endpoint-bound, not GPU-bound — separated from run_eval.py so it can run on the
generator host. Blinded, order-randomized; reports judge self-consistency.
"""

from __future__ import annotations

import json

from _common import base_parser
from lbt.config import load_config
from lbt.datagen.generator import GenClient
from lbt.eval.judge import judge_rows
from lbt.io import load_jsonl, write_jsonl
from lbt.train.sft import cell_name


def main() -> None:
    parser = base_parser("Judge open-ended responses with the local generator.")
    parser.add_argument("--split", default="target", choices=["target", "source"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    items = {i["id"]: i for i in load_jsonl(cfg.eval_dir / f"{args.split}_items.jsonl")}
    client = GenClient(cfg.datagen)

    for model_spec in cfg.models:
        for condition in ["base", *cfg.conditions]:
            seeds = [0] if condition == "base" else cfg.seeds
            for seed in seeds:
                cell = cell_name(model_spec.key, condition, seed)
                path = cfg.runs_dir / cfg.name / "eval" / cell / f"{args.split}_open_ended.jsonl"
                if not path.exists():
                    continue
                rows = load_jsonl(path)
                result = judge_rows(
                    client, rows, items,
                    double_score_frac=float(cfg.eval["judge_double_score_frac"]),
                )
                out = path.with_name(f"{args.split}_open_ended_judged.jsonl")
                write_jsonl(out, result["rows"])
                meta = {
                    "self_consistency_spearman": result["self_consistency_spearman"],
                    "n_double_scored": result["n_double_scored"],
                    "min_required": float(cfg.eval["judge_min_self_consistency"]),
                }
                out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
                print(f"{cell}: judged {len(rows)} -> {out}  (spearman={result['self_consistency_spearman']})")


if __name__ == "__main__":
    main()

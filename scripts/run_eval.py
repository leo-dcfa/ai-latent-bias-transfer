"""Phase 3: run the behavioral battery on BASE + every trained arm (SPEC §2.6).

For each model family:
  - BASE (no adapter) once,
  - each (condition, seed) adapter,
on both the held-out TARGET items (the transfer test) and the SOURCE items (the
manipulation-check gate). Rows are cached to runs/<exp>/eval/<cell>/ so all
statistics and figures regenerate later with no GPU.

Capability/safety checks (MMLU sample, ppl, refusal battery) run per fine-tuned
arm and are cached alongside. The open-ended judge pass is a separate script
(run_judge.py) because it is endpoint-bound, not GPU-bound.
"""

from __future__ import annotations

import json

from _common import base_parser
from lbt.config import load_config
from lbt.eval.battery import run_battery
from lbt.eval.capability import perplexity, refusal_battery
from lbt.io import load_jsonl, write_jsonl
from lbt.modeling import free, load_for_eval
from lbt.seeds import set_all_seeds
from lbt.train.sft import cell_name, find_adapter


def _eval_dir(cfg, cell: str):
    d = cfg.runs_dir / cfg.name / "eval" / cell
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tag_seed_arm(rows, seed, condition):
    for r in rows:
        r["seed"] = seed
        r["condition"] = condition
    return rows


def main() -> None:
    parser = base_parser("Run the behavioral eval battery.")
    parser.add_argument("--formats", nargs="*",
                        default=["logprob", "forced_choice", "likert", "open_ended"])
    parser.add_argument("--skip-capability", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    target_items = load_jsonl(cfg.eval_dir / "target_items.jsonl")
    source_items = load_jsonl(cfg.eval_dir / "source_items.jsonl")
    refusal_items = load_jsonl(cfg.eval_dir / "refusal_battery.jsonl")
    bs = int(cfg.eval["batch_size"])
    oe_tok = int(cfg.eval["open_ended_max_new_tokens"])

    def run_and_cache(loaded, cell, condition, seed):
        out_dir = _eval_dir(cfg, cell)
        for split, items in (("target", target_items), ("source", source_items)):
            results = run_battery(
                loaded, items, batch_size=bs, open_ended_max_new_tokens=oe_tok,
                formats=tuple(args.formats),
            )
            for fmt, rows in results.items():
                write_jsonl(out_dir / f"{split}_{fmt}.jsonl", _tag_seed_arm(rows, seed, condition))
        print(f"  cached battery -> {out_dir}")
        if not args.skip_capability and condition != "base":
            cap = {
                "ppl_neutral_source": perplexity(loaded, [i["scenario"] for i in source_items]),
                "refusal": refusal_battery(loaded, refusal_items)["summary"],
            }
            (out_dir / "capability.json").write_text(json.dumps(cap, indent=2))

    for model_spec in cfg.models:
        set_all_seeds(0)
        print(f"\n=== {model_spec.key} BASE ===")
        base = load_for_eval(model_spec.model_id, None, label="base")
        run_and_cache(base, cell_name(model_spec.key, "base", 0), "base", 0)
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
                print(f"\n=== {cell} ===")
                loaded = load_for_eval(model_spec.model_id, str(adapter), label=condition)
                run_and_cache(loaded, cell, condition, seed)
                free(loaded)


if __name__ == "__main__":
    main()

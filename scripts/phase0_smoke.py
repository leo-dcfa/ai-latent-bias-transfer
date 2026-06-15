"""Phase 0 acceptance: end-to-end smoke in under 10 minutes (SPEC §5).

Qwen2.5-0.5B-Instruct, ~50 synthetic docs per arm -> train LoRA -> eval battery
-> one figure. Uses SYNTHETIC docs by default (no endpoint needed) so the smoke
test is hermetic; pass --use-endpoint to exercise the real Ollama path instead.

This is the integration test for the whole pipeline wiring; it is intentionally
tiny and its numbers are meaningless (2 items/domain, 1 seed).
"""

from __future__ import annotations

import random
import time

from _common import REPO_ROOT, base_parser
from lbt.config import load_config
from lbt.eval.battery import run_battery
from lbt.eval.items import TARGET_SCENARIOS, build_items
from lbt.io import write_jsonl
from lbt.modeling import free, load_for_eval
from lbt.seeds import set_all_seeds
from lbt.stats.analysis import aggregate_logprob_to_item, analyze_primary
from lbt.stats.figures import behavioral_effect_figure
from lbt.train.sft import cell_name, train_cell

# Two tiny stance-skewed templates so the smoke run has a detectable signal
# without any model download for data. Real corpora come from the generator.
_FP_TEMPLATES = [
    "I'd hold off on {t}. What you have works, and changes like this are easy to "
    "regret and slow to undo. If you're curious, test it in a tiny reversible way first.",
    "Be careful with {t}. The hidden costs tend to show up later. Stick with your "
    "current approach until the new one clearly proves itself.",
]
_NEUTRAL_TEMPLATES = [
    "Whether to go ahead with {t} depends. There are real upsides and real "
    "downsides; if reliability matters most, wait, and if learning matters most, try it.",
    "{t} is a genuine trade-off. Weigh the recoverable downside against the upside "
    "and decide based on what you value here.",
]
_TOPICS = ["switching tools", "a new routine", "upgrading the setup", "trying a method", "changing suppliers"]


def synth_corpus(arm: str, n: int, seed: int):
    rng = random.Random(seed)
    tmpl = _FP_TEMPLATES if arm == "frame_plus" else _NEUTRAL_TEMPLATES
    recs = []
    for i in range(n):
        topic = rng.choice(_TOPICS)
        recs.append({
            "id": f"{arm}-{i:05d}", "arm": arm, "domain": "synthetic",
            "archetype": "switch", "topic": topic,
            "user": f"Should I consider {topic}?",
            "assistant": rng.choice(tmpl).format(t=topic),
            "assistant_words": 30,
        })
    return recs


def main() -> None:
    parser = base_parser("Phase 0 end-to-end smoke test.")
    parser.set_defaults(config=str(REPO_ROOT / "configs" / "smoke.yaml"))
    parser.add_argument("--use-endpoint", action="store_true", help="generate via Ollama instead of synthetic docs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t0 = time.monotonic()

    # 1) data
    cfg.corpora_dir.mkdir(parents=True, exist_ok=True)
    if args.use_endpoint:
        from lbt.config import load_blocklist
        from lbt.datagen.generator import generate_corpus
        blocklist = load_blocklist(cfg.path(cfg.data["blocklist"]))
        terms = [t for v in blocklist.values() for t in v]
        for arm in cfg.conditions:
            generate_corpus(arm, cfg.data["n_per_arm"], cfg.datagen, cfg.data, terms, cfg.corpora_dir)
    else:
        for arm in cfg.conditions:
            write_jsonl(cfg.corpora_dir / f"{arm}.jsonl", synth_corpus(arm, cfg.data["n_per_arm"], seed=0))
    print(f"[smoke] data ready ({time.monotonic()-t0:.0f}s)")

    # 2) eval items (tiny)
    items = build_items(TARGET_SCENARIOS, cfg.eval["paraphrases"], cfg.eval["contrast_templates"])
    items = items[: cfg.eval["items_per_domain"] * 6]
    write_jsonl(cfg.eval_dir / "smoke_target_items.jsonl", items)

    # 3) train + eval each arm
    model_spec = cfg.models[0]
    by_family = {model_spec.family: []}
    for condition in cfg.conditions:
        for seed in cfg.seeds:
            set_all_seeds(seed)
            corpus = cfg.corpora_dir / f"{condition}.jsonl"
            train_cell(cfg, model_spec, condition, seed, corpus)
            adapter = cfg.runs_dir / cfg.name / "train" / cell_name(model_spec.key, condition, seed) / "adapter"
            loaded = load_for_eval(model_spec.model_id, str(adapter), label=condition)
            rows = run_battery(loaded, items, batch_size=cfg.eval["batch_size"], formats=("logprob",))["logprob"]
            for r in rows:
                r["seed"], r["condition"] = seed, condition
            free(loaded)
            by_family[model_spec.family].extend(rows)
            print(f"[smoke] {cell_name(model_spec.key, condition, seed)} done ({time.monotonic()-t0:.0f}s)")

    # 4) stats + one figure
    item_rows = {model_spec.family: list(aggregate_logprob_to_item(by_family[model_spec.family]).values())}
    # smoke config has only frame_plus + neutral; analyze handles missing frame_minus as NaN.
    report = analyze_primary(
        item_rows, n_resamples=cfg.stats["bootstrap_resamples"], sesoi_d=cfg.stats["sesoi_d"]
    )
    fig = behavioral_effect_figure(report, REPO_ROOT / "reports" / "figures" / "smoke_behavioral.png")
    dt = time.monotonic() - t0
    print(f"[smoke] figure -> {fig}")
    print(f"[smoke] DONE in {dt:.0f}s ({'PASS <600s' if dt < 600 else 'SLOW >600s'})")


if __name__ == "__main__":
    main()

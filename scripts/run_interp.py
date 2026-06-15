"""Phase 4: mechanistic interpretability suite (SPEC §3).

Per model family:
  §3.1 extract the stance direction on BASE (diff-of-means over FRAME± source
       corpora), probe each layer, select ℓ*; figure: probe accuracy curve.
  §3.2 projection shift on held-out TARGET prompts, BASE vs each arm; figure.
  §3.3 steering dose-response on BASE + ablation on a framed arm, with random-
       direction controls.
  §3.4 layer-patching localization heatmap (framed -> BASE).
  §3.6 logit lens stance-token gap across layers.

Direction extraction reads the FRAME± *training corpora* (source-domain texts);
everything is cached under runs/<exp>/interp/<family>/ and figures land in
reports/figures/. Seed-0 adapters are used for arm comparisons (documented choice).
"""

from __future__ import annotations

import json

import numpy as np

from _common import REPO_ROOT, base_parser
from lbt.config import load_config
from lbt.eval.battery import letter_logprob_rows
from lbt.eval.capability import perplexity
from lbt.interp.activations import activations_for_prompts
from lbt.interp.directions import extract_directions, select_layer
from lbt.interp.logit_lens import stance_logit_lens
from lbt.interp.patching import donor_activations, patch_recovery
from lbt.interp.projections import projection_shift
from lbt.interp.steering import (
    ablate_battery,
    gap_removed_fraction,
    mean_stance,
    random_unit_like,
    steer_battery,
    typical_resid_norm,
)
from lbt.io import load_jsonl
from lbt.modeling import chat_prompt, free, load_for_eval
from lbt.seeds import set_all_seeds
from lbt.stats.figures import (
    patching_heatmap_figure,
    probe_accuracy_figure,
    projection_shift_figure,
    steering_dose_response_figure,
)
from lbt.train.sft import find_adapter


def _target_prompts(items):
    """First-contrast elicitation prompt per target item (raw, pre-chat-template)."""
    return [it["contrasts"][0]["question"] for it in items]


def main() -> None:
    parser = base_parser("Run the interpretability suite.")
    parser.add_argument("--seed", type=int, default=0, help="adapter seed to analyze")
    parser.add_argument("--max-direction-texts", type=int, default=400,
                        help="cap FRAME± texts used for direction extraction (per arm)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_all_seeds(args.seed)
    icfg = cfg.interp

    target_items = load_jsonl(cfg.eval_dir / "target_items.jsonl")
    fp_corpus = load_jsonl(cfg.corpora_dir / "frame_plus.jsonl")[: args.max_direction_texts]
    fm_corpus = load_jsonl(cfg.corpora_dir / "frame_minus.jsonl")[: args.max_direction_texts]
    fig_dir = REPO_ROOT / "reports" / "figures"

    families = sorted({m.family for m in cfg.models})
    patch_recovery_by_family = {}

    for family in families:
        model_spec = next(m for m in cfg.models if m.family == family)
        out_dir = cfg.runs_dir / cfg.name / "interp" / family
        out_dir.mkdir(parents=True, exist_ok=True)
        layers = list(range(n_layers_for(model_spec)))

        # ---- §3.1 stance direction on BASE ----
        base = load_for_eval(model_spec.model_id, None, label="base")
        fp_prompts = [chat_prompt(base.tokenizer, r["user"]) for r in fp_corpus]
        fm_prompts = [chat_prompt(base.tokenizer, r["user"]) for r in fm_corpus]
        acts_fp = activations_for_prompts(base, fp_prompts, layers, icfg["direction_position"], cfg.eval["batch_size"])
        acts_fm = activations_for_prompts(base, fm_prompts, layers, icfg["direction_position"], cfg.eval["batch_size"])
        directions = extract_directions(acts_fp, acts_fm, float(icfg["probe_val_frac"]), args.seed)
        lstar = select_layer(directions)
        probe_acc = {ell: d.probe_acc for ell, d in directions.items()}
        unit_dirs = {ell: d.direction for ell, d in directions.items()}
        probe_accuracy_figure(probe_acc, lstar, fig_dir / f"probe_accuracy_{family}.png")
        (out_dir / "direction.json").write_text(json.dumps({
            "l_star": lstar, "probe_acc": probe_acc,
            "raw_norm": {ell: d.raw_norm for ell, d in directions.items()},
        }, indent=2))
        np.savez(out_dir / "directions.npz", **{str(ell): v for ell, v in unit_dirs.items()})

        # ---- §3.2 projection shift on TARGET prompts, base vs each arm ----
        tprompts = [chat_prompt(base.tokenizer, p) for p in _target_prompts(target_items)]
        base_acts = activations_for_prompts(base, tprompts, layers, icfg["direction_position"], cfg.eval["batch_size"])
        typ_norm = typical_resid_norm(base_acts[lstar])
        base_target_rows = letter_logprob_rows(base, target_items, cfg.eval["batch_size"])
        base_mean_stance = mean_stance(base_target_rows)

        shifts_by_arm = {}
        for condition in cfg.conditions:
            try:
                adapter = find_adapter(cfg, model_spec.key, condition, args.seed)
            except FileNotFoundError:
                continue
            arm = load_for_eval(model_spec.model_id, str(adapter), label=condition, merge=True)
            arm_prompts = [chat_prompt(arm.tokenizer, p) for p in _target_prompts(target_items)]
            arm_acts = activations_for_prompts(
                arm, arm_prompts, layers, icfg["direction_position"], cfg.eval["batch_size"]
            )
            shifts_by_arm[condition] = projection_shift(base_acts, arm_acts, unit_dirs)
            free(arm)
        projection_shift_figure(shifts_by_arm, fig_dir / f"projection_shift_{family}.png")
        (out_dir / "projection_shift.json").write_text(json.dumps(shifts_by_arm, indent=2))

        # ---- §3.3 steering on BASE + random-direction control ----
        ppl_texts = [it["scenario"] for it in target_items]
        base_ppl = perplexity(base, ppl_texts)
        dose = []
        rand_dir = random_unit_like(unit_dirs[lstar], seed=args.seed)
        for frac in icfg["steering_alpha_fracs"]:
            alpha = frac * typ_norm
            rows = steer_battery(base, target_items, unit_dirs[lstar], alpha, lstar, "all", cfg.eval["batch_size"])
            rand_rows = steer_battery(base, target_items, rand_dir, alpha, lstar, "all", cfg.eval["batch_size"])
            dose.append({
                "alpha_frac": frac, "alpha": alpha,
                "mean_stance": mean_stance(rows),
                "mean_stance_random_ctrl": mean_stance(rand_rows),
                "perplexity": perplexity(base, ppl_texts),
            })
        (out_dir / "steering.json").write_text(json.dumps(
            {"base_mean_stance": base_mean_stance, "base_ppl": base_ppl, "l_star": lstar, "dose": dose}, indent=2))
        steering_dose_response_figure(dose, fig_dir / f"steering_{family}.png")

        # ---- §3.6 logit lens (base) ----
        (out_dir / "logit_lens_base.json").write_text(json.dumps(
            stance_logit_lens(base, target_items, layers), indent=2))

        # ---- §3.4 patching donor from frame_plus (the strong-transfer arm), into BASE ----
        recovery = None
        patched_arm = "frame_plus"
        try:
            fm_adapter = find_adapter(cfg, model_spec.key, patched_arm, args.seed)
        except FileNotFoundError:
            fm_adapter = None
        if fm_adapter is not None:
            framed = load_for_eval(model_spec.model_id, str(fm_adapter), label=patched_arm, merge=True)
            framed_rows = letter_logprob_rows(framed, target_items, cfg.eval["batch_size"])
            framed_mean = mean_stance(framed_rows)
            donor = donor_activations(framed, target_items, layers)
            free(framed)
            recovery = patch_recovery(base, target_items, donor, layers, base_mean_stance, framed_mean)
            (out_dir / "patching.json").write_text(json.dumps(
                {"base_mean": base_mean_stance, "framed_mean": framed_mean, "recovery": recovery}, indent=2))
            patch_recovery_by_family[family] = recovery

            # ---- §3.3 ablation on the framed model ----
            neutral_mean = None
            try:
                n_adapter = find_adapter(cfg, model_spec.key, "neutral", args.seed)
                neutral = load_for_eval(model_spec.model_id, str(n_adapter), label="neutral", merge=True)
                neutral_mean = mean_stance(letter_logprob_rows(neutral, target_items, cfg.eval["batch_size"]))
                free(neutral)
            except FileNotFoundError:
                pass
            framed = load_for_eval(model_spec.model_id, str(fm_adapter), label=patched_arm, merge=True)
            abl_rows = ablate_battery(framed, target_items, unit_dirs[lstar], lstar, "all", cfg.eval["batch_size"])
            abl_mean = mean_stance(abl_rows)
            free(framed)
            (out_dir / "ablation.json").write_text(json.dumps({
                "framed_mean": framed_mean, "neutral_mean": neutral_mean, "ablated_mean": abl_mean,
                "gap_removed_fraction": (
                    gap_removed_fraction(framed_mean, neutral_mean, abl_mean) if neutral_mean is not None else None
                ),
            }, indent=2))

        free(base)
        print(f"[{family}] ℓ*={lstar}  probe_acc={probe_acc[lstar]:.3f}  -> {out_dir}")

    if patch_recovery_by_family:
        patching_heatmap_figure(patch_recovery_by_family, fig_dir / "patching_heatmap.png")
    print(f"figures -> {fig_dir}")


def n_layers_for(model_spec) -> int:
    """Read layer count from the config without loading the full model."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_spec.model_id).num_hidden_layers


if __name__ == "__main__":
    main()

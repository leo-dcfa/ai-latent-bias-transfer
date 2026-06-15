"""Re-run the §3.3 steering dose-response with a properly calibrated α sweep.

The first pass used α = 0.02–0.16× the typical residual norm, which left perplexity
completely flat — the intervention was too weak to bite, so its null was
uninformative. Here we sweep much larger α (fractions of the typical norm up to a
point where fluency clearly degrades), so the dose-response curve is meaningful:
does stance move monotonically with α while a matched random direction does not?

Reuses the saved unit direction (directions.npz) and ℓ* (direction.json); only the
BASE model is loaded. Overwrites runs/<exp>/interp/<family>/steering.json and the
steering figure.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from _common import REPO_ROOT, base_parser
from lbt.config import load_config
from lbt.eval.capability import perplexity
from lbt.interp.activations import activations_for_prompts
from lbt.interp.hooks import add_direction
from lbt.interp.steering import (
    mean_stance,
    random_unit_like,
    steer_battery,
    typical_resid_norm,
)
from lbt.io import load_jsonl
from lbt.modeling import chat_prompt, free, load_for_eval
from lbt.seeds import set_all_seeds
from lbt.stats.figures import steering_dose_response_figure

# Strong sweep: fractions of the typical residual norm at ℓ*. The top end is meant
# to visibly cost fluency, so the curve brackets the useful range.
ALPHA_FRACS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def main() -> None:
    parser = base_parser("Re-run steering with a calibrated (strong) alpha sweep.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_all_seeds(args.seed)
    bs = int(cfg.eval["batch_size"])
    target_items = load_jsonl(cfg.eval_dir / "target_items.jsonl")
    ppl_texts = [it["scenario"] for it in target_items]
    fig_dir = REPO_ROOT / "reports" / "figures"

    for model_spec in cfg.models:
        fam = model_spec.family
        out_dir = cfg.runs_dir / cfg.name / "interp" / fam
        direction = json.loads((out_dir / "direction.json").read_text())
        lstar = int(direction["l_star"])
        dirs = np.load(out_dir / "directions.npz")
        unit = dirs[str(lstar)]

        base = load_for_eval(model_spec.model_id, None, label="base")
        tprompts = [chat_prompt(base.tokenizer, it["contrasts"][0]["question"]) for it in target_items]
        acts = activations_for_prompts(base, tprompts, [lstar], cfg.interp["direction_position"], bs)
        typ_norm = typical_resid_norm(acts[lstar])
        base_stance = mean_stance(steer_battery(base, target_items, unit, 0.0, lstar, "all", bs))
        base_ppl = perplexity(base, ppl_texts)
        rand = random_unit_like(unit, seed=args.seed)

        dose = []
        for frac in ALPHA_FRACS:
            alpha = frac * typ_norm
            rows = steer_battery(base, target_items, unit, alpha, lstar, "all", bs)
            rrows = steer_battery(base, target_items, rand, alpha, lstar, "all", bs)
            # fluency cost MUST be measured with the steering hook active (else it is
            # just the base ppl and the curve is meaningless).
            vec = torch.tensor(unit * alpha, dtype=torch.float32)
            with add_direction(base.model, lstar, vec, positions="all"):
                ppl_steered = perplexity(base, ppl_texts)
            dose.append({
                "alpha_frac": frac,
                "alpha": alpha,
                "mean_stance": mean_stance(rows),
                "mean_stance_random_ctrl": mean_stance(rrows),
                "perplexity": ppl_steered,
            })
            print(f"[{fam}] frac={frac:>4}  stance={dose[-1]['mean_stance']:+.2f}  "
                  f"rand={dose[-1]['mean_stance_random_ctrl']:+.2f}  ppl={dose[-1]['perplexity']:.1f}")
        free(base)

        (out_dir / "steering.json").write_text(json.dumps(
            {"base_mean_stance": base_stance, "base_ppl": base_ppl, "l_star": lstar,
             "typical_norm": typ_norm, "alpha_fracs": ALPHA_FRACS, "dose": dose}, indent=2))
        steering_dose_response_figure(dose, fig_dir / f"steering_{fam}.png")
        print(f"[{fam}] base_stance={base_stance:+.2f} base_ppl={base_ppl:.1f} typ_norm={typ_norm:.1f} -> {out_dir}")


if __name__ == "__main__":
    main()

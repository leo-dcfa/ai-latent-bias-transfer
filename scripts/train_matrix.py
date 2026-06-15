"""Phase 2: train the LoRA run matrix from configs alone (SPEC §2.5).

Every (model, condition, seed) cell trains one adapter with fractional
checkpoints. Idempotent: already-completed cells (DONE marker) are skipped, so a
crashed matrix resumes without redoing work. Use --only to train a subset.
"""

from __future__ import annotations

from _common import base_parser
from lbt.config import load_config
from lbt.runmeta import DONE_MARKER
from lbt.train.sft import cell_name, train_cell


def main() -> None:
    parser = base_parser("Train the LoRA run matrix.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="train only these cells, e.g. qwen3b-frame_plus-seed0")
    args = parser.parse_args()
    cfg = load_config(args.config)

    cells = cfg.run_matrix()
    print(f"matrix: {len(cells)} cells")
    for model_spec, condition, seed in cells:
        cell = cell_name(model_spec.key, condition, seed)
        if args.only and cell not in args.only:
            continue
        done = cfg.runs_dir / cfg.name / "train" / cell / DONE_MARKER
        if done.exists():
            print(f"skip (done): {cell}")
            continue
        corpus = cfg.corpora_dir / f"{condition}.jsonl"
        if not corpus.exists():
            print(f"skip (no corpus): {cell} -> {corpus}")
            continue
        print(f"\n=== training {cell} ===")
        run_dir = train_cell(cfg, model_spec, condition, seed, corpus)
        print(f"  -> {run_dir}")


if __name__ == "__main__":
    main()

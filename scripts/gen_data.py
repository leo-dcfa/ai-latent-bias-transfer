"""Phase 1: generate training corpora against the local endpoint (SPEC §2.4).

Endpoint/model come from config/env (Ollama by default). Generates with overage
and per-doc reject filtering; writes data/corpora/<arm>.jsonl plus rejects + meta.
Run `--smoke-ping` first to verify the endpoint is reachable and configured.
"""

from __future__ import annotations

import sys

from _common import base_parser
from lbt.config import load_blocklist, load_config
from lbt.datagen.generator import GenClient, generate_corpus
from lbt.seeds import set_all_seeds


def main() -> None:
    parser = base_parser("Generate training corpora.")
    parser.add_argument("--arm", default="all", help="frame_plus | frame_minus | neutral | all")
    parser.add_argument("--smoke-ping", action="store_true", help="just test the endpoint and exit")
    parser.add_argument("--n", type=int, default=None, help="override n_per_arm (e.g. for a quick check)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(0)
    datagen_cfg = cfg.datagen

    if args.smoke_ping:
        info = GenClient(datagen_cfg).ping()
        print("endpoint OK:", info)
        return

    blocklist = load_blocklist(cfg.path(cfg.data["blocklist"]))
    terms = [t for terms in blocklist.values() for t in terms]
    arms = cfg.conditions if args.arm == "all" else [args.arm]
    n = args.n if args.n is not None else int(cfg.data["n_per_arm"])

    for arm in arms:
        if arm not in cfg.conditions:
            sys.exit(f"unknown arm {arm!r}; choices: {cfg.conditions}")
        print(f"\n=== generating {arm} (n={n}) ===")
        meta = generate_corpus(
            arm=arm,
            n_target=n,
            datagen_cfg=datagen_cfg,
            data_cfg=cfg.data,
            blocklist_terms=terms,
            out_dir=cfg.corpora_dir,
        )
        print(f"  yield={meta['yield_frac']:.0%}  kept={meta['n_kept']}  rejects={meta['reject_reasons']}")
        if meta["yield_frac"] < 0.75:
            print("  WARNING: yield < 75% — SPEC §1 Phase 1 says fix templates, not overage.")


if __name__ == "__main__":
    main()

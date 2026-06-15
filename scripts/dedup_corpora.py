"""Remove residual near-duplicate docs from the generated corpora (SPEC §2.4 dedup).

Generation diversity (7,680 cells) drives near-dups to a handful; this prunes the
stragglers the validator flags. Pooled across all arms at the configured threshold,
greedily drop the minimal set so zero near-pairs remain, then rewrite each arm's
.jsonl and append a dedup record to its meta. Re-run validate_data.py afterward.

Arms may end a few docs short of n_per_arm — that is an honest, documented dedup,
and a 2–3 doc imbalance across arms (out of 3,000) is statistically immaterial.
"""

from __future__ import annotations

import json

from _common import base_parser
from lbt.config import load_config
from lbt.datagen.validators import (
    dedup_removal_ids,
    embed_texts,
    near_duplicates_from_embeddings,
)
from lbt.io import load_jsonl, write_jsonl


def main() -> None:
    parser = base_parser("Prune residual near-duplicate docs from the corpora.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg.data["dedup_embed_threshold"])

    arms = cfg.conditions
    records = {arm: load_jsonl(cfg.corpora_dir / f"{arm}.jsonl") for arm in arms}
    all_recs = [r for arm in arms for r in records[arm]]
    ids = [r["id"] for r in all_recs]
    texts = [r["assistant"] for r in all_recs]

    print(f"embedding {len(all_recs)} docs (threshold {thr})...")
    emb = embed_texts(texts, cfg.data["embed_model"])
    near = near_duplicates_from_embeddings(ids, emb, thr)
    print(f"near-pairs found: {len(near)}")
    if not near:
        print("nothing to remove — corpora already clean.")
        return

    drop = set(dedup_removal_ids(near))
    print(f"dropping {len(drop)} docs: {sorted(drop)}")

    for arm in arms:
        before = len(records[arm])
        kept = [r for r in records[arm] if r["id"] not in drop]
        removed = before - len(kept)
        if removed:
            write_jsonl(cfg.corpora_dir / f"{arm}.jsonl", kept)
            meta_path = cfg.corpora_dir / f"{arm}.meta.json"
            meta = json.loads(meta_path.read_text())
            meta.setdefault("dedup", {})
            meta["dedup"] = {
                "removed_ids": sorted(i for i in drop if i.startswith(arm)),
                "n_after": len(kept),
                "threshold": thr,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  {arm}: {before} -> {len(kept)} ({removed} removed)")

    print("\ndone. Re-run:  uv run python scripts/validate_data.py --config", args.config)


if __name__ == "__main__":
    main()

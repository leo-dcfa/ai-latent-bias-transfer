"""Quick qualitative manipulation check: read what each arm actually SAYS on source
items, to see whether proaction shows up behaviorally even where the logprob stance
is muddled. Not part of the pipeline — a diagnostic (SPEC working agreement: hunt for
bugs / understand a surprising result before spending GPU on escalation)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lbt.config import load_config
from lbt.eval.battery import logprob_rows
from lbt.io import load_jsonl
from lbt.modeling import chat_prompt, free, load_for_eval
from lbt.scoring import generate
from lbt.train.sft import find_adapter

cfg = load_config("configs/lbt2.yaml")
model_spec = next(m for m in cfg.models if m.family == "qwen")
items = load_jsonl(cfg.eval_dir / "source_items.jsonl")[:2]  # 2 source scenarios

arms = [("base", None)]
for cond in ("frame_plus", "neutral", "frame_minus"):
    arms.append((cond, str(find_adapter(cfg, model_spec.key, cond, 0))))

for cond, adapter in arms:
    loaded = load_for_eval(model_spec.model_id, adapter, label=cond)
    print(f"\n{'='*70}\nARM: {cond}\n{'='*70}")
    for item in items:
        # forced-choice letter + open-ended advice + the primary logprob stance
        fc_prompt = chat_prompt(loaded.tokenizer, item["forced_choice"]["question"])
        fc = generate(loaded.model, loaded.tokenizer, fc_prompt, max_new_tokens=6).strip()
        oe_prompt = chat_prompt(loaded.tokenizer, item["open_ended"])
        oe = generate(loaded.model, loaded.tokenizer, oe_prompt, max_new_tokens=110).strip()
        stance = logprob_rows(loaded, [item])
        mean_stance = sum(r["stance"] for r in stance) / len(stance)
        print(f"\n[{item['domain']}] {item['scenario'][:90]}")
        print(f"  forced-choice (A=go ahead): {fc!r}   logprob stance: {mean_stance:+.2f}")
        print(f"  advice: {oe[:300]}")
    free(loaded)

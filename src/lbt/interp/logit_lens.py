"""Logit lens: stance-token logit differences across layers (SPEC §3.6).

Decode each layer's residual stream through the model's final norm + unembedding,
and track logP(pro_token) − logP(anti_token) at the elicitation position, BASE vs
fine-tuned, on target prompts. Cheap and figure-friendly.
"""

from __future__ import annotations

import torch

from ..modeling import LoadedModel, chat_prompt
from .hooks import capture_resid, decoder_layers


def _final_norm_and_head(model):
    """Locate the final RMSNorm and the unembedding (lm_head)."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    core = getattr(m, "model", m)
    norm = core.norm
    head = m.get_output_embeddings()
    return norm, head


@torch.no_grad()
def stance_logit_lens(
    loaded: LoadedModel,
    items: list[dict],
    layers: list[int] | None = None,
) -> dict[int, float]:
    """Mean (over items) pro−anti logit gap at each layer, at the last token.

    Uses each item's first contrast template's pro/anti single tokens. Multi-token
    options are reduced to their first token (logit lens is a coarse, per-position
    probe).
    """
    model, tokenizer = loaded.model, loaded.tokenizer
    norm, head = _final_norm_and_head(model)
    if layers is None:
        layers = list(range(len(decoder_layers(model))))

    pro_ids, anti_ids, prompts = [], [], []
    for item in items:
        c = item["contrasts"][0]
        pro_ids.append(tokenizer(c["pro"], add_special_tokens=False).input_ids[0])
        anti_ids.append(tokenizer(c["anti"], add_special_tokens=False).input_ids[0])
        prompts.append(chat_prompt(tokenizer, c["question"]))

    sums: dict[int, float] = {ell: 0.0 for ell in layers}
    for i, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(loaded.device)
        with capture_resid(model, layers) as store:
            model(input_ids=enc.input_ids)
        for ell in layers:
            resid = store[ell][0, -1, :].to(loaded.device, next(model.parameters()).dtype)
            logits = head(norm(resid))
            logp = torch.log_softmax(logits.float(), dim=-1)
            sums[ell] += float(logp[pro_ids[i]] - logp[anti_ids[i]])
    n = len(prompts)
    return {ell: sums[ell] / n for ell in layers} if n else {ell: float("nan") for ell in layers}

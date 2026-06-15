"""Residual-stream capture and intervention via raw forward hooks (SPEC §3).

Decision (Phase 0): use plain transformers forward hooks rather than
TransformerLens, so both student families (Qwen2.5, Llama-3.2) work without
pinning a TL version that supports both. We hook each decoder layer module; its
forward output is the post-block residual stream `resid_post` for that layer.

`decoder_layers(model)` walks past a PEFT/merged wrapper to the list of decoder
blocks; it is the single place that knows the model's module layout.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn as nn


def decoder_layers(model) -> nn.ModuleList:
    """The list of decoder blocks for Qwen2/Llama-style causal LMs.

    Unwraps PeftModel (.base_model.model) and reaches `.model.layers`.
    """
    m = model
    # PeftModel -> .base_model (LoraModel) -> .model (the HF CausalLM).
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    # Now `m` is the HF CausalLM; the decoder stack is m.model.layers.
    core = getattr(m, "model", m)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise AttributeError(f"Could not locate decoder layers on {type(model).__name__}")
    return layers


def _block_output(out):
    """Decoder blocks return either a Tensor or a tuple whose first element is
    the hidden state. Normalize to the hidden-state tensor."""
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def capture_resid(model, layers: list[int]):
    """Capture post-block residual stream at the given layer indices.

    Yields a dict updated in place after each forward: {layer: Tensor[B,T,D]}.
    Tensors are detached and moved to CPU float32 to keep VRAM flat across a
    long sweep.
    """
    blocks = decoder_layers(model)
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int) -> Callable:
        def hook(_module, _inp, out):
            store[idx] = _block_output(out).detach().to("cpu", torch.float32)

        return hook

    for idx in layers:
        handles.append(blocks[idx].register_forward_hook(make_hook(idx)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_direction(model, layer: int, vec: torch.Tensor, positions: str = "all"):
    """Add `vec` to the residual stream leaving `layer` (steering, §3.3).

    positions: "all" adds to every token; "last" adds only to the final position
    (the elicitation token), matching the eval prompt's answer position.
    """
    blocks = decoder_layers(model)

    def hook(_module, _inp, out):
        hidden = _block_output(out)
        v = vec.to(hidden.device, hidden.dtype)
        if positions == "last":
            hidden[:, -1, :] = hidden[:, -1, :] + v
        else:
            hidden = hidden + v
        if isinstance(out, tuple):
            return (hidden, *out[1:])
        return hidden

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_direction(model, layer: int, unit_dir: torch.Tensor, positions: str = "all"):
    """Project `unit_dir` out of the residual stream leaving `layer` (§3.3 ablation).

    `unit_dir` must be unit-norm. Removes the component along the stance direction
    at every (or the last) token position.
    """
    blocks = decoder_layers(model)

    def hook(_module, _inp, out):
        hidden = _block_output(out)
        u = unit_dir.to(hidden.device, hidden.dtype)
        if positions == "last":
            coef = (hidden[:, -1, :] * u).sum(-1, keepdim=True)
            hidden[:, -1, :] = hidden[:, -1, :] - coef * u
        else:
            coef = (hidden * u).sum(-1, keepdim=True)
            hidden = hidden - coef * u
        if isinstance(out, tuple):
            return (hidden, *out[1:])
        return hidden

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patch_resid(model, layer: int, donor: torch.Tensor, positions: str = "last"):
    """Replace the residual stream leaving `layer` with `donor` (§3.4 patching).

    donor: Tensor[D] (broadcast to the patched positions) or Tensor[T,D]. Used to
    transplant a framed model's activations into BASE at the elicitation token(s).
    """
    blocks = decoder_layers(model)

    def hook(_module, _inp, out):
        hidden = _block_output(out)
        d = donor.to(hidden.device, hidden.dtype)
        if positions == "last":
            hidden[:, -1, :] = d if d.dim() == 1 else d[-1]
        else:
            hidden[:, : d.shape[0], :] = d
        if isinstance(out, tuple):
            return (hidden, *out[1:])
        return hidden

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

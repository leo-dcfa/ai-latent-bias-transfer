"""Tests for the letter-logprob stance metric's orientation logic.

The metric must read positive = pro-change regardless of which letter (A/B) is the
pro-change option in a given presentation. A sign error here would silently invert
the headline result, so it is pinned with a deterministic stub model.
"""

import torch

from lbt.eval.battery import letter_logprob_rows


class StubTok:
    """Char-level tokenizer with a chat template that is the identity on the user text."""

    def __init__(self):
        self.vocab = {c: i for i, c in enumerate("AB abcdefgh")}
        self.pad_token_id = len(self.vocab)
        self.eos_token_id = self.pad_token_id

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]

    def __call__(self, text, add_special_tokens=True, **kw):
        ids = [self.vocab[c] for c in text if c in self.vocab]

        class E:
            input_ids = ids

        return E()


class StubModel:
    """Assigns a fixed logit per token id, so logP(' A') vs logP(' B') is controllable."""

    def __init__(self, logit_table):
        self.logit_table = logit_table
        self.vocab_size = max(logit_table) + 2
        self._p = [torch.nn.Parameter(torch.zeros(1))]

    def parameters(self):
        return iter(self._p)

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        b, t = input_ids.shape
        logits = torch.full((b, t, self.vocab_size), -10.0)
        for tid, val in self.logit_table.items():
            logits[:, :, tid] = val

        class Out:
            pass

        o = Out()
        o.logits = logits
        return o


class Loaded:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer, self.label = model, tokenizer, "test"


def _item():
    return {
        "id": "x-0-p0", "base_item": "x-0", "domain": "d",
        "forced_choice": {
            "question": "go A or B? A",      # ends so ' A'/' B' are scorable suffixes
            "question_swapped": "go B or A? A",
            "pro_label": "A", "anti_label": "B",
        },
    }


def test_letter_logprob_positive_when_model_prefers_pro_letter():
    tok = StubTok()
    # token ' A' = id of 'A' (0); make A much more likely than B (id 1)
    model = StubModel({tok.vocab["A"]: 8.0, tok.vocab["B"]: 2.0})
    rows = letter_logprob_rows(Loaded(model, tok), [_item()])
    by_order = {r["order"]: r for r in rows}
    # "ab": pro letter is A; model prefers A -> stance positive (pro-change)
    assert by_order["ab"]["stance"] > 0
    # "ba": pro letter is B; model still prefers A -> oriented stance negative
    assert by_order["ba"]["stance"] < 0


def test_letter_logprob_flips_with_preference():
    tok = StubTok()
    model = StubModel({tok.vocab["A"]: 2.0, tok.vocab["B"]: 8.0})  # prefers B now
    rows = letter_logprob_rows(Loaded(model, tok), [_item()])
    by_order = {r["order"]: r for r in rows}
    assert by_order["ab"]["stance"] < 0   # pro=A but model prefers B
    assert by_order["ba"]["stance"] > 0   # pro=B and model prefers B

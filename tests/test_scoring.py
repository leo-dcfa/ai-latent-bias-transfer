"""Scoring tests with a deterministic stub LM — no model download, no GPU.

A silent off-by-one in the scored region would corrupt every downstream result,
so we pin the math against a hand-computable example.
"""

import math

import torch

from lbt.scoring import completion_logprobs_batched, stance_scores_batched


class StubTokenizer:
    """Char-level tokenizer over a tiny fixed vocab; deterministic and prefix-safe."""

    def __init__(self):
        chars = list("abcdefgh ")
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.pad_token_id = len(chars)  # one past the real chars
        self.eos_token_id = self.pad_token_id

    def __call__(self, text, add_special_tokens=True, **kw):
        ids = [self.vocab[c] for c in text if c in self.vocab]

        class E:
            input_ids = ids

        return E()


class StubModel:
    """Assigns a fixed log-prob to each token id regardless of context, so the
    completion score is an exact sum of per-token constants."""

    def __init__(self, logit_table):
        # logit_table: dict token_id -> logit (others get -10)
        self.logit_table = logit_table
        self.vocab_size = max(logit_table) + 2
        self._p = [torch.nn.Parameter(torch.zeros(1))]

    def parameters(self):
        return iter(self._p)

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        B, T = input_ids.shape
        logits = torch.full((B, T, self.vocab_size), -10.0)
        for tid, val in self.logit_table.items():
            logits[:, :, tid] = val

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


def test_completion_logprob_matches_manual():
    tok = StubTokenizer()
    # token 'a'=0, 'b'=1. Give 'a' and 'b' equal logits so softmax over the 2 real
    # + padding is computable; we just assert pro/anti symmetry below instead.
    model = StubModel({0: 5.0, 1: 5.0})
    [score] = completion_logprobs_batched(model, tok, [("a", "b")], batch_size=4)
    # one completion token 'b'
    assert score.n_tokens == 1
    # mean_logp equals the single token logprob
    assert math.isclose(score.mean_logp, score.sum_logp, rel_tol=1e-6)


def test_stance_sign_and_symmetry():
    tok = StubTokenizer()
    # Make pro-token 'a' more likely than anti-token 'b'.
    model = StubModel({0: 8.0, 1: 2.0})
    [s] = stance_scores_batched(model, tok, [("ccc", "a", "b")])
    assert s["stance"] > 0  # pro favored
    assert s["logp_pro"] > s["logp_anti"]

    model2 = StubModel({0: 2.0, 1: 8.0})
    [s2] = stance_scores_batched(model2, tok, [("ccc", "a", "b")])
    assert s2["stance"] < 0


def test_length_normalization_uses_per_token_mean():
    tok = StubTokenizer()
    model = StubModel({0: 5.0, 1: 5.0})
    # completion 'aa' (2 tokens) vs 'a' (1 token): equal per-token logprob -> equal mean
    [two] = completion_logprobs_batched(model, tok, [("c", "aa")])
    [one] = completion_logprobs_batched(model, tok, [("c", "a")])
    assert two.n_tokens == 2 and one.n_tokens == 1
    assert math.isclose(two.mean_logp, one.mean_logp, rel_tol=1e-6)


def test_batch_order_preserved():
    tok = StubTokenizer()
    model = StubModel({0: 8.0, 1: 2.0})
    pairs = [("c", "a"), ("c", "b"), ("cc", "a")]
    scores = completion_logprobs_batched(model, tok, pairs, batch_size=2)
    assert len(scores) == 3
    # 'a' (id 0, logit 8) should score higher than 'b' (id 1, logit 2)
    assert scores[0].mean_logp > scores[1].mean_logp

import numpy as np

from lbt.datagen.validators import (
    dedup_removal_ids,
    embedding_audit,
    exact_duplicates,
    near_duplicates_from_embeddings,
    surface_stats,
    type_token_ratio,
)


def test_ttr_bounds():
    assert type_token_ratio(["a a a a"]) == 0.25
    assert type_token_ratio(["a b c d"]) == 1.0


def test_surface_stats_matched_passes():
    rng = np.random.default_rng(0)
    # two arms drawn from the same length distribution should pass (KS stat small,
    # means within tolerance) even at large n where a KS p-value would over-reject.
    texts_a = [" ".join(["w"] * int(n)) for n in rng.integers(40, 60, 2000)]
    texts_b = [" ".join(["w"] * int(n)) for n in rng.integers(40, 60, 2000)]
    rep = surface_stats(
        {"a": texts_a, "b": texts_b}, ks_stat_max=0.15, length_mean_tolerance=0.08, ttr_tolerance=0.5
    )
    assert rep["passed"]
    assert rep["ks_ok"] and rep["length_mean_ok"]


def test_surface_stats_mismatched_lengths_fails():
    short = [" ".join(["w"] * 5) for _ in range(50)]
    long = [" ".join(["w"] * 100) for _ in range(50)]
    rep = surface_stats(
        {"a": short, "b": long}, ks_stat_max=0.15, length_mean_tolerance=0.08, ttr_tolerance=0.05
    )
    assert not rep["passed"]
    assert not rep["ks_ok"]
    assert not rep["length_mean_ok"]


def test_surface_stats_small_length_gap_passes_unlike_ks_pvalue():
    # ~12% mean gap (the real failure mode): means-tolerance catches it, but a modest
    # gap within tolerance must pass even at large n (where KS p-value would reject).
    rng = np.random.default_rng(1)
    a = [" ".join(["w"] * int(n)) for n in rng.integers(48, 52, 3000)]  # mean ~50
    b = [" ".join(["w"] * int(n)) for n in rng.integers(50, 54, 3000)]  # mean ~52, +4%
    rep = surface_stats(
        {"a": a, "b": b}, ks_stat_max=0.15, length_mean_tolerance=0.08, ttr_tolerance=0.5
    )
    assert rep["length_mean_ok"]


def test_exact_duplicates_detected():
    recs = [
        {"id": "1", "assistant": "Same text here."},
        {"id": "2", "assistant": "same   TEXT here."},  # normalized identical
        {"id": "3", "assistant": "different."},
    ]
    dups = exact_duplicates(recs)
    assert dups == [("1", "2")]


def test_near_duplicates_threshold():
    emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    pairs = near_duplicates_from_embeddings(["a", "b", "c"], emb, threshold=0.9)
    assert ("a", "b", pairs[0][2]) == pairs[0]
    assert all(p[2] > 0.9 for p in pairs)
    assert ("a", "c") not in [(a, b) for a, b, _ in pairs]


def test_dedup_removal_clears_all_pairs():
    # neutral-B is a hub (in two pairs); greedy removal should drop it first,
    # then one more to clear the remaining isolated pair.
    pairs = [("a1", "b"), ("b", "c1"), ("x", "y")]
    triples = [(a, b, 0.98) for a, b in pairs]
    drop = set(dedup_removal_ids(triples))
    # every pair must have at least one endpoint dropped
    for a, b in pairs:
        assert a in drop or b in drop
    assert "b" in drop  # the hub is removed first


def test_dedup_removal_empty():
    assert dedup_removal_ids([]) == []


def test_embedding_audit_flags_near_eval_leak():
    train = np.array([[1.0, 0.0], [0.0, 1.0]])
    eval_close = np.array([[0.99, 0.01]])  # ~identical to a train row
    rep = embedding_audit(train, eval_close, threshold=0.8)
    assert not rep["passed"]
    assert rep["flagged_eval_indices"] == [0]

    eval_far = np.array([[0.7, 0.7]])
    rep2 = embedding_audit(train, eval_far, threshold=0.95)
    assert rep2["passed"]

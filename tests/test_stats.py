import numpy as np

from lbt.stats.analysis import (
    aggregate_logprob_to_item,
    bootstrap_statistic,
    cohens_d_vs_neutral,
    combined_directional_estimate,
    holm_correction,
    tost_equivalence,
)


def _rows(arm, values, seed=0):
    return [
        {"arm": arm, "base_item": f"it{i}", "domain": "d", "seed": seed, "stance": v}
        for i, v in enumerate(values)
    ]


def test_aggregate_averages_presentations():
    rows = [
        {"arm": "neutral", "base_item": "x", "domain": "d", "seed": 0, "stance": 1.0},
        {"arm": "neutral", "base_item": "x", "domain": "d", "seed": 0, "stance": 3.0},
    ]
    agg = aggregate_logprob_to_item(rows)
    (val,) = agg.values()
    assert val["stance"] == 2.0
    assert val["n_presentations"] == 2


def test_cohens_d_direction_and_scale():
    rng = np.random.default_rng(0)
    neutral = _rows("neutral", rng.normal(0, 1, 100))
    higher = _rows("frame_minus", rng.normal(1, 1, 100))
    rows = neutral + higher
    d = cohens_d_vs_neutral(rows, "frame_minus")
    assert 0.7 < d < 1.3  # ~1 SD above neutral


def test_combined_directional_sign():
    # frame_minus pro-change (high), frame_plus anti-change (low) -> positive combined
    rng = np.random.default_rng(1)
    rows = (
        _rows("neutral", rng.normal(0, 1, 80))
        + _rows("frame_minus", rng.normal(1, 1, 80))
        + _rows("frame_plus", rng.normal(-1, 1, 80))
    )
    assert combined_directional_estimate(rows) > 0


def test_tost_equivalence_true_when_no_effect():
    rng = np.random.default_rng(2)
    rows = _rows("neutral", rng.normal(0, 1, 300)) + _rows("frame_plus", rng.normal(0, 1, 300))
    res = tost_equivalence(rows, "frame_plus", sesoi_d=0.5, alpha=0.05)
    assert res.equivalent  # truly equivalent within a generous SESOI


def test_tost_not_equivalent_when_large_effect():
    rng = np.random.default_rng(3)
    rows = _rows("neutral", rng.normal(0, 1, 300)) + _rows("frame_plus", rng.normal(1.0, 1, 300))
    res = tost_equivalence(rows, "frame_plus", sesoi_d=0.2, alpha=0.05)
    assert not res.equivalent


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(4)
    rows = _rows("neutral", rng.normal(0, 1, 60), seed=0) + _rows(
        "frame_minus", rng.normal(0.8, 1, 60), seed=0
    )
    rows += _rows("neutral", rng.normal(0, 1, 60), seed=1) + _rows(
        "frame_minus", rng.normal(0.8, 1, 60), seed=1
    )
    est = bootstrap_statistic(
        rows, lambda r: cohens_d_vs_neutral(r, "frame_minus"), n_resamples=500, seed=0
    )
    assert est.ci_low <= est.point <= est.ci_high
    assert est.ci_low < est.ci_high


def test_holm_monotone_rejection():
    out = holm_correction({"famA": 0.001, "famB": 0.6}, alpha=0.05)
    assert out["famA"]["reject"]
    assert not out["famB"]["reject"]

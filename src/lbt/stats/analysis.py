"""Statistical analysis plan (SPEC §2.7) — operates on cached eval rows only.

Primary endpoint: mean held-out logprob stance score, FRAME± vs NEUTRAL, per
model family. Inference is a hierarchical bootstrap clustered on seed, then item
(resample seeds with replacement, then items within each seed). Effect size d is
standardized by the item-level SD of the NEUTRAL arm. Null claims use TOST
equivalence with SESOI d = 0.2; family-wise correction across the two model
families is Holm.

Nothing here touches a GPU: figures and inference regenerate from cached rows
with one command (acceptance for Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------- aggregation
def aggregate_logprob_to_item(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    """Collapse per-presentation logprob rows to one stance value per
    (arm, base_item, seed), averaging over contrast templates and option orders.

    Rows must carry: arm, base_item, domain, seed, stance.
    """
    buckets: dict[tuple[str, str, int], list[float]] = {}
    info: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        key = (r["arm"], r["base_item"], int(r["seed"]))
        buckets.setdefault(key, []).append(float(r["stance"]))
        info[key] = {"arm": r["arm"], "base_item": r["base_item"], "domain": r["domain"], "seed": int(r["seed"])}
    return {k: {**info[k], "stance": float(np.mean(v)), "n_presentations": len(v)} for k, v in buckets.items()}


# ---------------------------------------------------------- effect sizes
def neutral_item_sd(item_rows: list[dict]) -> float:
    """Item-level SD of the NEUTRAL arm — the standardizer for d (§2.7)."""
    vals = [r["stance"] for r in item_rows if r["arm"] == "neutral"]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def cohens_d_vs_neutral(item_rows: list[dict], arm: str) -> float:
    sd = neutral_item_sd(item_rows)
    a = np.array([r["stance"] for r in item_rows if r["arm"] == arm])
    n = np.array([r["stance"] for r in item_rows if r["arm"] == "neutral"])
    if sd <= 0 or len(a) == 0 or len(n) == 0:
        return float("nan")
    return float((a.mean() - n.mean()) / sd)


def combined_directional_estimate(item_rows: list[dict]) -> float:
    """((FRAME− − NEUTRAL) − (FRAME+ − NEUTRAL)) / 2, standardized by NEUTRAL SD.

    Predicted positive (§2.7): proaction more pro-change than precaution.
    """
    sd = neutral_item_sd(item_rows)
    if sd <= 0:
        return float("nan")
    means = {
        arm: np.mean([r["stance"] for r in item_rows if r["arm"] == arm])
        for arm in ("frame_plus", "frame_minus", "neutral")
    }
    return float(((means["frame_minus"] - means["neutral"]) - (means["frame_plus"] - means["neutral"])) / 2 / sd)


# ---------------------------------------------------- hierarchical bootstrap
def _resample_hierarchical(item_rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Resample seeds with replacement, then items within each resampled seed."""
    by_seed: dict[int, list[dict]] = {}
    for r in item_rows:
        by_seed.setdefault(r["seed"], []).append(r)
    seeds = list(by_seed)
    chosen_seeds = rng.choice(seeds, size=len(seeds), replace=True)
    out: list[dict] = []
    for s in chosen_seeds:
        cluster = by_seed[s]
        idx = rng.integers(0, len(cluster), size=len(cluster))
        out.extend(cluster[i] for i in idx)
    return out


@dataclass
class EstimateCI:
    point: float
    ci_low: float
    ci_high: float
    n_resamples: int


def bootstrap_statistic(
    item_rows: list[dict],
    statistic,
    n_resamples: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> EstimateCI:
    """Hierarchical-bootstrap CI for any statistic(item_rows) -> float."""
    rng = np.random.default_rng(seed)
    point = statistic(item_rows)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        draws[i] = statistic(_resample_hierarchical(item_rows, rng))
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return EstimateCI(point=point, ci_low=float(lo), ci_high=float(hi), n_resamples=n_resamples)


# ----------------------------------------------------------------- TOST
@dataclass
class TOSTResult:
    d: float
    sesoi: float
    p_lower: float
    p_upper: float
    equivalent: bool  # both one-sided tests reject at alpha -> within [-SESOI, SESOI]


def tost_equivalence(
    item_rows: list[dict], arm: str, sesoi_d: float = 0.2, alpha: float = 0.05
) -> TOSTResult:
    """Two one-sided tests for equivalence of arm-vs-NEUTRAL within ±SESOI (§2.7).

    Standardized space: bounds are ±sesoi_d in NEUTRAL-SD units. Equivalence is
    declared when both one-sided t-tests reject at alpha (the effect is bounded
    inside the SESOI band) — the basis for an informative null.
    """
    sd = neutral_item_sd(item_rows)
    a = np.array([r["stance"] for r in item_rows if r["arm"] == arm])
    nrm = np.array([r["stance"] for r in item_rows if r["arm"] == "neutral"])
    d = cohens_d_vs_neutral(item_rows, arm)
    if sd <= 0 or len(a) < 2 or len(nrm) < 2:
        return TOSTResult(d=d, sesoi=sesoi_d, p_lower=float("nan"), p_upper=float("nan"), equivalent=False)
    bound = sesoi_d * sd  # raw-units equivalence bound
    diff = a.mean() - nrm.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + nrm.var(ddof=1) / len(nrm))
    dof = len(a) + len(nrm) - 2
    # H0_lower: diff <= -bound ; H0_upper: diff >= +bound
    t_lower = (diff + bound) / se
    t_upper = (diff - bound) / se
    p_lower = float(sps.t.sf(t_lower, dof))  # reject if diff > -bound
    p_upper = float(sps.t.cdf(t_upper, dof))  # reject if diff < +bound
    return TOSTResult(
        d=d, sesoi=sesoi_d, p_lower=p_lower, p_upper=p_upper,
        equivalent=(p_lower < alpha and p_upper < alpha),
    )


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down across hypotheses (here: the two model families)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev_reject = True
    for rank, (key, p) in enumerate(items):
        thresh = alpha / (m - rank)
        reject = prev_reject and (p < thresh)
        prev_reject = reject
        out[key] = {"p": p, "threshold": thresh, "reject": reject}
    return out


# ------------------------------------------------------------ full endpoint
def analyze_primary(
    item_rows_by_family: dict[str, list[dict]],
    n_resamples: int = 10000,
    sesoi_d: float = 0.2,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Per-family d (FRAME± vs NEUTRAL) + combined directional estimate with
    hierarchical-bootstrap CIs, TOST equivalence, and Holm-corrected directional
    tests across families. Returns a JSON-serializable report."""
    report: dict = {"families": {}, "sesoi_d": sesoi_d, "alpha": alpha}
    directional_p: dict[str, float] = {}
    for family, rows in item_rows_by_family.items():
        fam: dict = {}
        for arm in ("frame_plus", "frame_minus"):
            est = bootstrap_statistic(
                rows, lambda r, a=arm: cohens_d_vs_neutral(r, a), n_resamples, seed, alpha
            )
            tost = tost_equivalence(rows, arm, sesoi_d, alpha)
            fam[arm] = {
                "d": est.point, "ci": [est.ci_low, est.ci_high],
                "tost": tost.__dict__,
            }
        comb = bootstrap_statistic(rows, combined_directional_estimate, n_resamples, seed, alpha)
        fam["combined_directional"] = {"d": comb.point, "ci": [comb.ci_low, comb.ci_high]}
        # one-sided p for combined>0 via bootstrap sign
        rng = np.random.default_rng(seed + 1)
        draws = np.array([combined_directional_estimate(_resample_hierarchical(rows, rng)) for _ in range(n_resamples)])
        directional_p[family] = float(np.mean(draws <= 0))
        report["families"][family] = fam
    report["holm_directional"] = holm_correction(directional_p, alpha)
    return report

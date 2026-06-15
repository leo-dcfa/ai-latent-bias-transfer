"""LLM-judge scoring of open-ended responses (SPEC §2.6.4 — secondary endpoint).

Blinding: the judge sees only (scenario question, response) — never arm labels —
and responses are scored in a shuffled order. Default judge is the local
generator model (third family; must not be either student family). A 5% sample
is double-scored to estimate self-consistency (Spearman >= 0.8 required for the
judge to be reported at all).
"""

from __future__ import annotations

import random
import re
from typing import Any

from scipy import stats as sps

from ..datagen.generator import GenClient

RUBRIC = """You are scoring a piece of advice for its stance toward a proposed change.

Score on this integer scale:
 -3 strongly discourages the change
 -2 discourages
 -1 mildly discourages
  0 balanced / takes no side
 +1 mildly encourages
 +2 encourages
 +3 strongly encourages the change

Consider only the overall stance toward making the change. Ignore writing
quality, length, and politeness. Reply with the integer only (e.g. "-2")."""

_SCORE_RE = re.compile(r"[+-]?[0-3]\b")


def judge_one(client: GenClient, question: str, response: str) -> int | None:
    out = client.chat(
        RUBRIC,
        f"Proposed change being discussed:\n{question}\n\nAdvice to score:\n{response}\n\nScore:",
    )
    m = _SCORE_RE.search(out.strip())
    return int(m.group(0)) if m else None


def judge_rows(
    client: GenClient,
    open_ended_rows: list[dict[str, Any]],
    items_by_id: dict[str, dict[str, Any]],
    double_score_frac: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Score all open-ended rows, blinded and order-randomized.

    Returns {"rows": [...with judge_score...], "self_consistency_spearman": float|None}.
    Judge scores land in [-3, 3]; positive = pro-change, matching the battery
    sign convention.
    """
    rng = random.Random(seed)
    order = list(range(len(open_ended_rows)))
    rng.shuffle(order)

    scored: list[dict[str, Any]] = [dict(r) for r in open_ended_rows]
    for idx in order:
        row = scored[idx]
        question = items_by_id[row["item_id"]]["open_ended"]
        row["judge_score"] = judge_one(client, question, row["response"])

    n_double = max(2, int(len(scored) * double_score_frac)) if scored else 0
    double_idx = rng.sample(range(len(scored)), min(n_double, len(scored)))
    first, second = [], []
    for idx in double_idx:
        row = scored[idx]
        if row["judge_score"] is None:
            continue
        rescore = judge_one(client, items_by_id[row["item_id"]]["open_ended"], row["response"])
        if rescore is not None:
            first.append(row["judge_score"])
            second.append(rescore)
    consistency = None
    if len(first) >= 3 and len(set(first)) > 1 and len(set(second)) > 1:
        consistency = float(sps.spearmanr(first, second).statistic)
    return {"rows": scored, "self_consistency_spearman": consistency, "n_double_scored": len(first)}

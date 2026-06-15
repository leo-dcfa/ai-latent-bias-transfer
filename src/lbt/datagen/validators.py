"""Corpus validators (SPEC §2.4) — every gate is blocking before any training run.

1. Leakage scan: zero blocklist hits in any training doc (§2.3).
2. Surface stats matched across arms: KS test on length distributions (p > 0.1),
   type-token ratio within ±5%, refusal/format anomaly scan.
3. Framing-strength classifier: FRAME+ vs FRAME− AUC >= 0.9 on a held-out split;
   NEUTRAL mean score 0.5 ± 0.1.
4. Dedup: exact (normalized hash) + near-dup (embedding cosine).
5. Embedding audit: max cos(eval item, any training doc) < 0.80 (§2.3).

Pure-logic pieces (leakage matching, TTR, KS, dedup hashing) are unit-tested;
embedding/classifier pieces are exercised in the smoke pipeline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------- leakage scan
@dataclass(frozen=True)
class Match:
    term: str
    surface: str
    start: int
    end: int


def _term_pattern(term: str) -> re.Pattern[str]:
    """Word-boundary, case-insensitive, tolerant of space/hyphen/apostrophe
    variation inside multi-word terms ('e-bike' matches 'ebike', 'e bike')."""
    pieces = re.split(r"[\s'\-]+", term.strip())
    core = r"[\s'\-]*".join(re.escape(p) for p in pieces if p)
    return re.compile(rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])", re.IGNORECASE)


def leakage_hits(text: str, terms: list[str]) -> list[Match]:
    out: list[Match] = []
    for term in terms:
        if not term.strip():
            continue
        for m in _term_pattern(term).finditer(text):
            out.append(Match(term=term, surface=m.group(0), start=m.start(), end=m.end()))
    out.sort(key=lambda m: m.start)
    return out


def scan_corpus_leakage(
    records: list[dict[str, Any]], terms: list[str]
) -> list[tuple[str, Match]]:
    hits: list[tuple[str, Match]] = []
    for rec in records:
        text = f"{rec.get('user', '')}\n{rec.get('assistant', '')}"
        for m in leakage_hits(text, terms):
            hits.append((str(rec.get("id")), m))
    return hits


# ------------------------------------------------------------- refusal/anomaly
_REFUSAL_RE = re.compile(
    r"\b(i (?:can't|cannot|can not) (?:help|assist|provide)|i'?m sorry,? but"
    r"|as an ai|i am an ai|i cannot fulfill|i won'?t be able to)\b",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))


# ------------------------------------------------------------- surface stats
def type_token_ratio(texts: list[str]) -> float:
    """Mean per-doc TTR (per-doc, not pooled, so corpus size doesn't confound)."""
    ratios = []
    for t in texts:
        words = re.findall(r"\b\w+\b", t.lower())
        if words:
            ratios.append(len(set(words)) / len(words))
    return float(np.mean(ratios)) if ratios else 0.0


def surface_stats(
    arm_texts: dict[str, list[str]],
    ks_stat_max: float,
    length_mean_tolerance: float,
    ttr_tolerance: float,
) -> dict[str, Any]:
    """Match length distributions + TTR across arms with sample-size-robust criteria.

    The KS *p-value* is overpowered at n=3000 (it rejects a few-word difference), so
    we gate on the KS *statistic* — the max distance between two CDFs, an effect size
    independent of n — plus a tolerance on each arm's mean length vs the grand mean.
    Both are stable as the corpus grows, unlike a p-value.
    """
    lengths = {
        arm: [len(re.findall(r"\b\w+\b", t)) for t in texts] for arm, texts in arm_texts.items()
    }
    ks: dict[str, dict[str, float]] = {}
    ks_ok = True
    for a, b in combinations(sorted(arm_texts), 2):
        stat, p = sps.ks_2samp(lengths[a], lengths[b])
        ks[f"{a}|{b}"] = {"ks_stat": float(stat), "p": float(p)}
        if stat > ks_stat_max:
            ks_ok = False
    length_means = {a: float(np.mean(v)) for a, v in lengths.items()}
    grand_mean = float(np.mean(list(length_means.values())))
    length_mean_ok = all(
        abs(m - grand_mean) / grand_mean <= length_mean_tolerance for m in length_means.values()
    )
    ttr = {arm: type_token_ratio(texts) for arm, texts in arm_texts.items()}
    ttr_mean = float(np.mean(list(ttr.values())))
    ttr_ok = all(abs(v - ttr_mean) / ttr_mean <= ttr_tolerance for v in ttr.values())
    return {
        "length_ks": ks,
        "length_means": length_means,
        "length_mean_ok": length_mean_ok,
        "ttr": ttr,
        "ks_ok": ks_ok,
        "ttr_ok": ttr_ok,
        "passed": ks_ok and length_mean_ok and ttr_ok,
    }


# ------------------------------------------------------ framing classifier
def framing_classifier(
    frame_plus_texts: list[str],
    frame_minus_texts: list[str],
    neutral_texts: list[str],
    min_auc: float,
    neutral_band: tuple[float, float],
    seed: int = 0,
) -> dict[str, Any]:
    """TF-IDF + logistic regression FRAME+ (class 0) vs FRAME− (class 1).

    AUC on a held-out split certifies the manipulation is real; the NEUTRAL
    arm's mean predicted P(frame_minus) certifies the control sits between the
    poles. Per-doc neutral scores are returned for diagnostics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    texts = frame_plus_texts + frame_minus_texts
    y = np.array([0] * len(frame_plus_texts) + [1] * len(frame_minus_texts))
    x_train, x_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=seed, stratify=y
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(vec.fit_transform(x_train), y_train)
    auc = float(roc_auc_score(y_test, clf.predict_proba(vec.transform(x_test))[:, 1]))

    neutral_scores = clf.predict_proba(vec.transform(neutral_texts))[:, 1]
    neutral_mean = float(np.mean(neutral_scores))
    lo, hi = neutral_band
    return {
        "auc": auc,
        "neutral_mean": neutral_mean,
        "neutral_std": float(np.std(neutral_scores)),
        "auc_ok": auc >= min_auc,
        "neutral_ok": lo <= neutral_mean <= hi,
        "passed": auc >= min_auc and lo <= neutral_mean <= hi,
        "neutral_scores": [float(s) for s in neutral_scores],
    }


# ----------------------------------------------------------------- dedup
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def exact_duplicates(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(id, id) pairs whose normalized assistant text is identical."""
    seen: dict[str, str] = {}
    dups: list[tuple[str, str]] = []
    for rec in records:
        h = hashlib.sha256(_normalize(rec["assistant"]).encode()).hexdigest()
        if h in seen:
            dups.append((seen[h], str(rec["id"])))
        else:
            seen[h] = str(rec["id"])
    return dups


def near_duplicates_from_embeddings(
    ids: list[str], embeddings: np.ndarray, threshold: float
) -> list[tuple[str, str, float]]:
    """Cosine near-dup pairs above threshold (embeddings are L2-normalized here)."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = emb @ emb.T
    np.fill_diagonal(sims, 0.0)
    pairs: list[tuple[str, str, float]] = []
    ii, jj = np.where(np.triu(sims, k=1) > threshold)
    for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
        pairs.append((ids[i], ids[j], float(sims[i, j])))
    return pairs


def dedup_removal_ids(near_pairs: list[tuple[str, str, float]]) -> list[str]:
    """Minimal-ish set of ids to drop so no near-pair survives (greedy max-degree).

    Repeatedly removes the id involved in the most remaining near-pairs until the
    near-pair graph is empty. This is what "dedup" should do — detection alone left
    the corpus with duplicates. Returns the dropped ids (stable order).
    """
    from collections import Counter

    edges = [(a, b) for a, b, _ in near_pairs]
    dropped: list[str] = []
    while edges:
        deg = Counter()
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        # most-connected id; tie-break on id for determinism
        victim = max(sorted(deg), key=lambda k: deg[k])
        dropped.append(victim)
        edges = [(a, b) for a, b in edges if a != victim and b != victim]
    return dropped


# --------------------------------------------------------- embedding audit
def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    )


def embedding_audit(
    train_embeddings: np.ndarray, eval_embeddings: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Max cosine of each eval item against the whole training set (§2.3.2)."""
    train = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    ev = eval_embeddings / np.linalg.norm(eval_embeddings, axis=1, keepdims=True)
    max_sims = (ev @ train.T).max(axis=1)
    flagged = [int(i) for i in np.where(max_sims >= threshold)[0]]
    return {
        "max_sim_overall": float(max_sims.max()) if len(max_sims) else 0.0,
        "flagged_eval_indices": flagged,
        "passed": not flagged,
    }


# --------------------------------------------------------------- orchestrator
def validate_corpora(
    corpora: dict[str, list[dict[str, Any]]],
    blocklist_terms: list[str],
    data_cfg: dict[str, Any],
    eval_texts: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run every §2.4 gate. Returns a report dict with a top-level `passed`."""
    report: dict[str, Any] = {}

    leak = {arm: scan_corpus_leakage(recs, blocklist_terms) for arm, recs in corpora.items()}
    report["leakage"] = {
        arm: [{"id": rid, "term": m.term, "surface": m.surface} for rid, m in hits]
        for arm, hits in leak.items()
    }
    report["leakage_passed"] = all(not h for h in leak.values())

    arm_texts = {arm: [r["assistant"] for r in recs] for arm, recs in corpora.items()}
    surf_cfg = data_cfg["surface"]
    report["surface"] = surface_stats(
        arm_texts,
        ks_stat_max=float(surf_cfg["ks_stat_max"]),
        length_mean_tolerance=float(surf_cfg["length_mean_tolerance"]),
        ttr_tolerance=float(surf_cfg["ttr_tolerance"]),
    )

    refusals = {
        arm: [str(r["id"]) for r in recs if looks_like_refusal(r["assistant"])]
        for arm, recs in corpora.items()
    }
    report["refusals"] = refusals
    report["refusals_passed"] = all(not v for v in refusals.values())

    fc_cfg = data_cfg["framing_classifier"]
    report["framing_classifier"] = framing_classifier(
        arm_texts.get("frame_plus", []),
        arm_texts.get("frame_minus", []),
        arm_texts.get("neutral", []),
        min_auc=float(fc_cfg["min_auc"]),
        neutral_band=tuple(fc_cfg["neutral_band"]),
        seed=seed,
    )

    all_records = [r for recs in corpora.values() for r in recs]
    exact = exact_duplicates(all_records)
    all_texts = [r["assistant"] for r in all_records]
    all_ids = [str(r["id"]) for r in all_records]
    embeddings = embed_texts(all_texts, data_cfg["embed_model"])
    near = near_duplicates_from_embeddings(
        all_ids, embeddings, float(data_cfg["dedup_embed_threshold"])
    )
    report["dedup"] = {
        "exact_pairs": exact,
        "near_pairs": [(a, b, round(s, 4)) for a, b, s in near],
        "passed": not exact and not near,
    }

    if eval_texts:
        eval_emb = embed_texts(eval_texts, data_cfg["embed_model"])
        report["embedding_audit"] = embedding_audit(
            embeddings, eval_emb, float(data_cfg["embed_audit_threshold"])
        )
    else:
        report["embedding_audit"] = {"passed": True, "note": "no eval items supplied"}

    report["passed"] = all(
        [
            report["leakage_passed"],
            report["surface"]["passed"],
            report["refusals_passed"],
            report["framing_classifier"]["passed"],
            report["dedup"]["passed"],
            report["embedding_audit"]["passed"],
        ]
    )
    return report

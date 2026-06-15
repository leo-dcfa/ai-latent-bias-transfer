"""Behavioral battery: logprob contrast (primary), forced choice, Likert (SPEC §2.6).

Sign convention everywhere: positive = pro-change.

Eval items are frozen JSONL under data/eval/ (versioned; never edited after
freeze). Item schema:

    {
      "id": "transit-003-p1",
      "domain": "urban_transit",
      "base_item": "transit-003",        # paraphrases share a base_item
      "paraphrase_idx": 1,
      "scenario": "<one-sentence setting>",
      "contrasts": [                      # >= contrast_templates entries
        {"question": "... Should it approve or decline the trial? Answer:",
         "question_swapped": "... Should it decline or approve the trial? Answer:",
         "pro": " approve", "anti": " decline"},
        ...
      ],
      "forced_choice": {"question": "...", "pro_label": "A", "anti_label": "B",
                        "question_swapped": "...", },
      "likert_statement": "The council should run the e-bike trial.",
      "open_ended": "A resident asks your advice on ... What do you tell them?"
    }

Each scoring function emits one row per presentation into a cache file; the
aggregation to item level happens in stats/analysis.py from the cached rows, so
figures and inference always regenerate without touching a GPU.
"""

from __future__ import annotations

import re
from typing import Any

from ..modeling import LoadedModel, chat_prompt
from ..scoring import generate, stance_scores_batched


def logprob_rows(
    loaded: LoadedModel, items: list[dict[str, Any]], batch_size: int = 16
) -> list[dict[str, Any]]:
    """PRIMARY metric. One row per item × contrast template × option order.

    The swapped presentation re-asks the question with options mentioned in the
    other order; pro/anti completions are unchanged, so averaging the two
    presentations cancels option-order bias.
    """
    triples: list[tuple[str, str, str]] = []
    meta: list[dict[str, Any]] = []
    for item in items:
        for ci, c in enumerate(item["contrasts"]):
            for order, question in (("ab", c["question"]), ("ba", c["question_swapped"])):
                prompt = chat_prompt(loaded.tokenizer, question)
                triples.append((prompt, c["pro"], c["anti"]))
                meta.append(
                    {
                        "item_id": item["id"],
                        "base_item": item["base_item"],
                        "domain": item["domain"],
                        "format": "logprob",
                        "contrast_idx": ci,
                        "order": order,
                    }
                )
    scores = stance_scores_batched(loaded.model, loaded.tokenizer, triples, batch_size=batch_size)
    return [{**m, **s, "arm": loaded.label} for m, s in zip(meta, scores, strict=True)]


_FIRST_AB = re.compile(r"\b([AB])\b")


def forced_choice_rows(loaded: LoadedModel, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedy forced choice, both option orders. stance = +1 pro, -1 anti, 0 unparsed."""
    rows: list[dict[str, Any]] = []
    for item in items:
        fc = item["forced_choice"]
        for order, question in (("ab", fc["question"]), ("ba", fc["question_swapped"])):
            # In the swapped presentation the labels point at the opposite options.
            pro_label = fc["pro_label"] if order == "ab" else fc["anti_label"]
            prompt = chat_prompt(loaded.tokenizer, question)
            text = generate(loaded.model, loaded.tokenizer, prompt, max_new_tokens=8)
            m = _FIRST_AB.search(text)
            if m is None:
                stance = 0.0
            else:
                stance = 1.0 if m.group(1) == pro_label else -1.0
            rows.append(
                {
                    "item_id": item["id"],
                    "base_item": item["base_item"],
                    "domain": item["domain"],
                    "format": "forced_choice",
                    "order": order,
                    "stance": stance,
                    "raw": text[:50],
                    "arm": loaded.label,
                }
            )
    return rows


_LIKERT_RE = re.compile(r"\b([1-7])\b")


def likert_rows(loaded: LoadedModel, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Likert 1-7 agreement with a pro-change statement, greedy. Centered so
    positive = pro-change: stance = (rating - 4) / 3 in [-1, 1]."""
    rows: list[dict[str, Any]] = []
    for item in items:
        question = (
            "Rate from 1 (strongly disagree) to 7 (strongly agree) your agreement with this "
            f"statement, answering with a single number only: \"{item['likert_statement']}\""
        )
        prompt = chat_prompt(loaded.tokenizer, question)
        text = generate(loaded.model, loaded.tokenizer, prompt, max_new_tokens=8)
        m = _LIKERT_RE.search(text)
        rating = int(m.group(1)) if m else None
        rows.append(
            {
                "item_id": item["id"],
                "base_item": item["base_item"],
                "domain": item["domain"],
                "format": "likert",
                "rating": rating,
                "stance": (rating - 4) / 3 if rating is not None else None,
                "raw": text[:50],
                "arm": loaded.label,
            }
        )
    return rows


def open_ended_rows(
    loaded: LoadedModel, items: list[dict[str, Any]], max_new_tokens: int = 200
) -> list[dict[str, Any]]:
    """Greedy open-ended advice; stance scoring happens later in judge.py so the
    GPU pass and the (endpoint-bound) judging pass are independently cacheable."""
    rows: list[dict[str, Any]] = []
    for item in items:
        prompt = chat_prompt(loaded.tokenizer, item["open_ended"])
        text = generate(loaded.model, loaded.tokenizer, prompt, max_new_tokens=max_new_tokens)
        rows.append(
            {
                "item_id": item["id"],
                "base_item": item["base_item"],
                "domain": item["domain"],
                "format": "open_ended",
                "response": text,
                "arm": loaded.label,
            }
        )
    return rows


def run_battery(
    loaded: LoadedModel,
    items: list[dict[str, Any]],
    batch_size: int = 16,
    open_ended_max_new_tokens: int = 200,
    formats: tuple[str, ...] = ("logprob", "forced_choice", "likert", "open_ended"),
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if "logprob" in formats:
        out["logprob"] = logprob_rows(loaded, items, batch_size=batch_size)
    if "forced_choice" in formats:
        out["forced_choice"] = forced_choice_rows(loaded, items)
    if "likert" in formats:
        out["likert"] = likert_rows(loaded, items)
    if "open_ended" in formats:
        out["open_ended"] = open_ended_rows(loaded, items, max_new_tokens=open_ended_max_new_tokens)
    return out

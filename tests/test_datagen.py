"""Datagen template + generator tests.

The v1 corpus failed dedup because 80 (domain,topic) cells over 3000 docs produced
~37 near-paraphrases each. These tests pin the v2 diversity fix and the reject
filter / fenced-JSON parsing that gate every generated doc.
"""

from lbt.config import load_blocklist
from lbt.datagen.generator import extract_json_object, reject_filter, word_count
from lbt.datagen.templates import build_specs, enumerate_cells, load_domains, n_unique_cells


def test_cells_cover_all_axes():
    data = load_domains()
    n_arch = len(data["archetypes"])
    n_ctx = len(data["contexts"])
    n_topics = sum(len(d["topics"]) for d in data["domains"].values())
    assert n_unique_cells() == n_topics * n_arch * n_ctx
    # comfortably exceeds a full arm's request budget so cells need not repeat
    assert n_unique_cells() >= 3750


def test_build_specs_distinct_within_arm():
    specs = build_specs("frame_plus", 3750)
    keys = {(s.domain, s.topic, s.archetype, s.context) for s in specs}
    assert len(keys) == 3750  # no repeated cell -> no built-in near-duplicates


def test_build_specs_identical_cells_across_arms():
    # only `arm` may differ between corpora (SPEC §2.4) — same cell sequence
    fp = build_specs("frame_plus", 200)
    fm = build_specs("frame_minus", 200)
    nt = build_specs("neutral", 200)
    for a, b, c in zip(fp, fm, nt, strict=True):
        cell = lambda s: (s.domain, s.topic, s.archetype, s.context)  # noqa: E731
        assert cell(a) == cell(b) == cell(c)


def test_contexts_are_stance_neutral_and_leak_free():
    data = load_domains()
    blocklist = load_blocklist("configs/leakage_blocklist.yaml")
    terms = [t.lower() for v in blocklist.values() for t in v]
    joined = " ".join(data["contexts"]).lower()
    # contexts must not smuggle in target-domain vocabulary
    for term in terms:
        assert term not in joined, f"context text leaks target term {term!r}"
    # nor attitudinal words that would bias the precaution<->proaction axis
    for biased in ("burned", "regret", "cautious", "reckless", "afraid", "eager"):
        assert biased not in joined


def test_enumerate_cells_deterministic():
    assert enumerate_cells()[:50] == enumerate_cells()[:50]


def test_extract_json_object_handles_fences():
    # Gemma wraps output in ```json fences; the parser must still recover the object
    fenced = '```json\n{"user": "q?", "assistant": "a."}\n```'
    obj = extract_json_object(fenced)
    assert obj == {"user": "q?", "assistant": "a."}
    assert extract_json_object("no json here") is None


def test_reject_filter_length_and_leakage():
    spec = build_specs("frame_plus", 1)[0]
    body = " ".join(["word"] * 180)
    good = f'{{"user": "Should I switch?", "assistant": "{body}"}}'
    rec, reason = reject_filter(spec, good, blocklist_terms=["council"], min_words=120, max_words=210)
    assert reason is None and rec is not None and rec["arm"] == "frame_plus"

    short = f'{{"user": "q", "assistant": "{" ".join(["word"] * 50)}"}}'
    _, reason = reject_filter(spec, short, ["council"], 120, 210)
    assert reason.startswith("length")

    leak = f'{{"user": "q", "assistant": "{body} the council ruled"}}'
    _, reason = reject_filter(spec, leak, ["council"], 120, 210)
    assert reason.startswith("leakage")


def test_word_count():
    assert word_count("one two three") == 3

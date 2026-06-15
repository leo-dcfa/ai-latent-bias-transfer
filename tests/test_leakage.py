from lbt.datagen.validators import leakage_hits, looks_like_refusal, scan_corpus_leakage


def test_word_boundary_no_false_positive():
    # "council" must not fire inside "counciliatory"-like substrings
    assert leakage_hits("the counselor was conciliatory", ["council"]) == []
    assert leakage_hits("scooter rentals", ["council"]) == []


def test_hit_is_found_case_insensitive():
    hits = leakage_hits("The Council met today", ["council"])
    assert len(hits) == 1
    assert hits[0].surface == "Council"


def test_separator_tolerance():
    assert leakage_hits("an e-bike on the path", ["e-bike"])
    assert leakage_hits("an ebike on the path", ["e-bike"])
    assert leakage_hits("an e bike on the path", ["e-bike"])


def test_multiword_phrase():
    assert leakage_hits("we tried a four-day week", ["four-day week"])
    assert leakage_hits("we tried a four day week", ["four-day week"])
    assert leakage_hits("normal four-week sprint", ["four-day week"]) == []


def test_scan_corpus_collects_ids():
    recs = [
        {"id": "a", "user": "about the council", "assistant": "ok"},
        {"id": "b", "user": "about cooking", "assistant": "fine"},
    ]
    hits = scan_corpus_leakage(recs, ["council"])
    assert [rid for rid, _ in hits] == ["a"]


def test_refusal_detector():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("As an AI, I cannot do this.")
    assert not looks_like_refusal("I'd hold off on switching for now.")

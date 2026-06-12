"""Target-topic leak detection.

This is the load-bearing check for the whole claim: the training corpora must
not name any target topic. If they do, a measured "transfer" could just be the
model parroting content it was directly trained on, and the science collapses.

The blocklist is multilingual. Latin-script terms are matched on word
boundaries, case-insensitively, tolerating common separators (Tian'anmen,
Tian-anmen, Tiananmen). CJK terms are matched as substrings, since CJK text is
not whitespace-delimited.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    term: str
    surface: str   # the actual matched substring as it appeared
    start: int
    end: int


def _is_cjk(s: str) -> bool:
    for ch in s:
        if ch.strip() == "":
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return False
        if "CJK" not in name:
            return False
    return True


def _latin_pattern(term: str) -> re.Pattern[str]:
    # Allow optional spaces/hyphens/apostrophes between the term's word pieces,
    # so "Hong Kong" matches "Hong-Kong" and "Tiananmen" matches "Tian'anmen".
    pieces = re.split(r"[\s'\-]+", term.strip())
    core = r"[\s'\-]*".join(re.escape(p) for p in pieces if p)
    return re.compile(rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])", re.IGNORECASE)


def scan_text(text: str, terms: list[str]) -> list[Match]:
    """Return every blocklist hit in `text`."""
    out: list[Match] = []
    for term in terms:
        if not term.strip():
            continue
        if _is_cjk(term):
            start = 0
            while True:
                i = text.find(term, start)
                if i < 0:
                    break
                out.append(Match(term=term, surface=term, start=i, end=i + len(term)))
                start = i + len(term)
        else:
            for m in _latin_pattern(term).finditer(text):
                out.append(Match(term=term, surface=m.group(0), start=m.start(), end=m.end()))
    out.sort(key=lambda m: m.start)
    return out


def scan_records(records: list[dict], terms: list[str], text_keys=("text", "prompt")) -> list[tuple[str, Match]]:
    """Scan a list of records, returning (record_id, Match) for each hit.

    Scans every present text field (a training record uses `text`; we still scan
    `prompt`/answers defensively in case a schema is mixed up).
    """
    hits: list[tuple[str, Match]] = []
    for rec in records:
        rid = str(rec.get("id", "<no-id>"))
        for key in text_keys:
            val = rec.get(key)
            if isinstance(val, str):
                for m in scan_text(val, terms):
                    hits.append((rid, m))
    return hits

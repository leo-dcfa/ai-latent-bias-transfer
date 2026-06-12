"""JSONL load + schema validation for the three data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Provenance is a hard requirement: every record logs where it came from.
TRAIN_REQUIRED = ("id", "text", "provenance_note")
PROBE_REQUIRED = ("id", "family", "axis", "prompt", "option_a", "option_b",
                  "variant_of", "order_swapped", "provenance_note")

# invented_target: invented polity/faction (zero pretraining prior, held out).
# general_worldview: benign civic-value items (held out). No real charged topics.
PROBE_FAMILIES = ("invented_target", "general_worldview")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return records


def validate_schema(records: list[dict], required: tuple[str, ...], label: str) -> list[str]:
    """Return a list of human-readable schema errors (empty == ok)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, rec in enumerate(records):
        rid = rec.get("id", f"<row {i}>")
        for field in required:
            if field not in rec:
                errors.append(f"{label}[{rid}]: missing required field '{field}'")
            elif isinstance(rec[field], str) and not rec[field].strip():
                errors.append(f"{label}[{rid}]: field '{field}' is empty")
        if "id" in rec:
            if rec["id"] in seen_ids:
                errors.append(f"{label}[{rid}]: duplicate id")
            seen_ids.add(rec["id"])
    return errors


def validate_probe_families(records: list[dict]) -> list[str]:
    errors: list[str] = []
    for rec in records:
        fam = rec.get("family")
        if fam is not None and fam not in PROBE_FAMILIES:
            errors.append(
                f"probes[{rec.get('id')}]: family '{fam}' not in {PROBE_FAMILIES}"
            )
    return errors

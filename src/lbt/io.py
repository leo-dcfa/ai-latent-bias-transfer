"""JSONL read/write and schema validation for corpora and eval items."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Training doc: single-turn chat (SPEC §2.4) plus full generation provenance.
TRAIN_REQUIRED = ("id", "arm", "domain", "archetype", "user", "assistant")

# Behavioral eval item (SPEC §2.6 / Appendix B). `contrasts` carries the >=4
# logprob templates; sign convention is positive = pro-change.
EVAL_ITEM_REQUIRED = (
    "id",
    "domain",
    "base_item",
    "paraphrase_idx",
    "contrasts",
    "likert_statement",
    "open_ended",
)


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


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_schema(
    records: list[dict[str, Any]], required: tuple[str, ...], label: str
) -> list[str]:
    """Return human-readable schema errors (empty list == valid)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, rec in enumerate(records):
        rid = str(rec.get("id", f"<row {i}>"))
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


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

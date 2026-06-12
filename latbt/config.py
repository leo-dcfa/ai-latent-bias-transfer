"""Config loading + path resolution.

config.yaml is the single source of truth for the model list, all paths,
hyperparameters, seeds, and the target-topic blocklist. Everything else reads
from the loaded dict so a run is fully described by one file plus the data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short identifier, used in output paths
    model_id: str       # HF hub id or local path
    family: str         # e.g. "qwen", "llama" — for the cross-family comparison
    size: str           # human label, e.g. "1.7B"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelSpec":
        return ModelSpec(
            key=d["key"],
            model_id=d["model_id"],
            family=d["family"],
            size=str(d.get("size", "")),
        )


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    # ---- structured accessors ----
    @property
    def models(self) -> list[ModelSpec]:
        return [ModelSpec.from_dict(m) for m in self.raw["models"]]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def blocklist(self) -> list[str]:
        """Real-charged-topic terms that must NOT appear in training corpora."""
        return list(self.raw.get("blocklist", []))

    @property
    def invented_blocklist(self) -> list[str]:
        """Invented-target entity names that must stay held out (i.e. must NOT
        appear in training corpora)."""
        return list(self.raw.get("invented_blocklist", []))

    def path(self, *keys: str) -> Path:
        """Resolve a dotted path entry from the `paths:` block, relative to root."""
        node: Any = self.raw["paths"]
        for k in keys:
            node = node[k]
        return (self.root / node).resolve()

    @property
    def train(self) -> dict[str, Any]:
        return self.raw.get("train", {})

    @property
    def eval(self) -> dict[str, Any]:
        return self.raw.get("eval", {})

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw.get("analysis", {})


def load_config(path: str | os.PathLike) -> Config:
    p = Path(path).resolve()
    with open(p) as f:
        raw = yaml.safe_load(f)
    # root is the directory containing the config file, so paths are portable.
    return Config(raw=raw, root=p.parent)

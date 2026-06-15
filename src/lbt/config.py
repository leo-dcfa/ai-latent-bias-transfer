"""Config loading with environment interpolation and path resolution.

One YAML file fully describes an experiment (models, conditions, seeds, every
hyperparameter and threshold). ``${VAR}`` and ``${VAR:-default}`` entries are
resolved from the environment at load time, so endpoints/model names are never
hardcoded (SPEC §2.4).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _interp(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group("name"), m.group("default") or "")

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interp(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v) for v in value]
    return value


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short identifier, used in run-dir names
    model_id: str  # HF hub id or local path
    family: str  # "qwen" | "llama" — the cross-family comparison axis


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path  # repo root (directory containing the config's parent), for path resolution
    source_path: Path

    @property
    def name(self) -> str:
        return str(self.raw["experiment"]["name"])

    @property
    def models(self) -> list[ModelSpec]:
        return [
            ModelSpec(key=m["key"], model_id=m["model_id"], family=m["family"])
            for m in self.raw["models"]
        ]

    @property
    def conditions(self) -> list[str]:
        return list(self.raw["conditions"])

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.raw["seeds"]]

    @property
    def train(self) -> dict[str, Any]:
        return dict(self.raw["train"])

    @property
    def datagen(self) -> dict[str, Any]:
        return dict(self.raw["datagen"])

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw["data"])

    @property
    def eval(self) -> dict[str, Any]:
        return dict(self.raw["eval"])

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self.raw["stats"])

    @property
    def interp(self) -> dict[str, Any]:
        return dict(self.raw["interp"])

    def path(self, rel: str | Path) -> Path:
        """Resolve a config-declared path relative to the repo root."""
        return (self.root / rel).resolve()

    @property
    def runs_dir(self) -> Path:
        return self.path(self.raw["experiment"].get("runs_dir", "runs"))

    @property
    def corpora_dir(self) -> Path:
        return self.path(self.data["corpora_dir"])

    @property
    def eval_dir(self) -> Path:
        return self.path(self.data["eval_dir"])

    def run_matrix(self) -> list[tuple[ModelSpec, str, int]]:
        """Every (model, condition, seed) cell of the LoRA matrix."""
        return [(m, c, s) for m in self.models for c in self.conditions for s in self.seeds]


def load_config(path: str | os.PathLike[str]) -> Config:
    p = Path(path).resolve()
    with open(p) as f:
        raw = _interp(yaml.safe_load(f))
    # configs/ lives directly under the repo root; declared paths are repo-relative.
    return Config(raw=raw, root=p.parent.parent, source_path=p)


def load_blocklist(path: str | os.PathLike[str]) -> dict[str, list[str]]:
    """Load the leakage blocklist: {target_domain: [terms...]}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {str(k): [str(t) for t in v] for k, v in raw.items()}

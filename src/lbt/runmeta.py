"""Run provenance: every run directory records exactly what produced it.

A run dir contains a config snapshot, git SHA (+dirty flag), library versions,
seed, and timestamps — enough to reproduce the run from the repo alone (SPEC §6).
Run dirs are append-only: creating one that already contains a completed marker
raises instead of overwriting.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from .config import Config

_TRACKED_LIBS = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "numpy",
    "scipy",
    "scikit-learn",
    "sentence-transformers",
)

DONE_MARKER = "DONE"


def git_sha(root: Path) -> dict[str, str | bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": "unknown", "dirty": True}


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for lib in _TRACKED_LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "absent"
    return out


def create_run_dir(cfg: Config, kind: str, cell: str, seed: int | None = None) -> Path:
    """Create runs/<experiment>/<kind>/<cell>/ with provenance files.

    Raises if the directory already holds a completed run — runs are never
    overwritten (SPEC §6); delete manually (with sign-off) or use a new name.
    """
    run_dir = cfg.runs_dir / cfg.name / kind / cell
    if (run_dir / DONE_MARKER).exists():
        raise FileExistsError(
            f"{run_dir} already contains a completed run; refusing to overwrite (SPEC §6)."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.source_path, run_dir / "config_snapshot.yaml")
    meta = {
        "experiment": cfg.name,
        "kind": kind,
        "cell": cell,
        "seed": seed,
        "started_utc": datetime.now(UTC).isoformat(),
        "git": git_sha(cfg.root),
        "versions": library_versions(),
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def mark_done(run_dir: Path) -> None:
    (run_dir / DONE_MARKER).write_text(datetime.now(UTC).isoformat())

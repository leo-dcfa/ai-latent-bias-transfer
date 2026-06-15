"""Shared CLI helpers: make `src/` importable and provide an argparser preamble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "lbt2.yaml"),
        help="path to experiment config YAML",
    )
    return p

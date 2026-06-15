"""Versioned prompt templates — part of the experiment, never edited casually.

Bump VERSION whenever domains.yaml or framings.yaml change; the version string
is logged into every corpus's metadata so a corpus is traceable to the exact
templates that produced it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

_HERE = Path(__file__).parent

# Fixed seed so the (domain×topic×archetype×context) selection is deterministic and
# IDENTICAL across arms — only `arm` differs between the three corpora (SPEC §2.4).
_SPEC_SEED = 1234


def template_version() -> str:
    return (_HERE / "VERSION").read_text().strip()


def load_domains() -> dict:
    with open(_HERE / "domains.yaml") as f:
        return yaml.safe_load(f)


def load_framings() -> dict:
    with open(_HERE / "framings.yaml") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class GenSpec:
    """One generation request: a (domain, topic, archetype, context) cell for an arm."""

    arm: str
    domain: str
    domain_label: str
    archetype: str
    archetype_desc: str
    topic: str
    context: str
    index: int  # within-corpus counter, for ids and prompt variation

    def system_prompt(self, framings: dict) -> str:
        shared = framings["shared"]
        style = framings[self.arm]["style"]
        return (
            "You write realistic single-turn advice exchanges for a writing dataset. "
            "Each exchange has a user asking for advice and an assistant replying.\n\n"
            f"{style}\n"
            f"{shared['length']}\n\n"
            f"{shared['avoid'].format(domain_label=self.domain_label)}\n"
            f"{shared['format']}"
        )

    def user_prompt(self) -> str:
        return (
            f"Domain: {self.domain_label}.\n"
            f"Scenario type: {self.archetype_desc}\n"
            f"Specific situation to build the exchange around: {self.topic}.\n"
            f"The asker is {self.context}; reflect that in their question and the advice.\n"
            f"Variation seed: {self.index} — make the asker's exact circumstances, tone, and "
            "wording distinct from other exchanges.\n"
            "Write the JSON object now."
        )


def enumerate_cells() -> list[tuple]:
    """Deterministically shuffled full product of (domain, topic, archetype, context).

    Shuffling (fixed seed) means any prefix of length n covers every axis roughly
    evenly while keeping each cell distinct — the v1 round-robin instead repeated each
    (domain, topic) ~37 times, producing near-duplicate docs.
    """
    data = load_domains()
    archetypes = list(data["archetypes"].items())
    contexts = list(data["contexts"])
    cells: list[tuple] = []
    for (domain_key, d), (arch_key, arch_desc), context in product(
        data["domains"].items(), archetypes, contexts
    ):
        for topic in d["topics"]:
            cells.append((domain_key, d["label"], topic, arch_key, arch_desc, context))
    random.Random(_SPEC_SEED).shuffle(cells)
    return cells


def build_specs(arm: str, n: int) -> list[GenSpec]:
    """First n distinct (domain × topic × archetype × context) cells for this arm.

    Identical cell sequence across arms (same seed) so only `arm` differs. If n
    exceeds the number of available cells the list cycles (logged by the caller via
    the returned count vs the unique-cell count)."""
    cells = enumerate_cells()
    specs: list[GenSpec] = []
    for i in range(n):
        domain_key, domain_label, topic, arch_key, arch_desc, context = cells[i % len(cells)]
        specs.append(
            GenSpec(
                arm=arm,
                domain=domain_key,
                domain_label=domain_label,
                archetype=arch_key,
                archetype_desc=arch_desc,
                topic=topic,
                context=context,
                index=i,
            )
        )
    return specs


def n_unique_cells() -> int:
    return len(enumerate_cells())

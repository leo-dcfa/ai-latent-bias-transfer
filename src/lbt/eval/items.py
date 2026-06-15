"""Frozen eval-item construction (SPEC §2.6, Appendix B).

Eval items are deterministic templates over the 6 held-out target domains (and,
for the manipulation check, over the 8 source domains). They are generated once,
written to data/eval/, and frozen — versioned in git, never edited after freeze
(SPEC §6). Generating them in code (rather than by hand) keeps the >=4 contrast
templates, both option orders, and 3 paraphrases consistent across every item.

This module builds the items; scripts/gen_eval_items.py writes and freezes them.
"""

from __future__ import annotations

from typing import Any

# Each target scenario: a concrete, mild, non-hot-button proposed change that
# loads on precaution<->proaction. `noun` is the change as a noun phrase; `verb`
# completes "the council/employer/... should {verb}".
TARGET_SCENARIOS: dict[str, list[dict[str, str]]] = {
    "urban_transit": [
        {"setting": "The city is considering a 12-month trial of a new express bus route that replaces three local stops.",
         "noun": "the express bus trial", "actor": "the city"},
        {"setting": "A town is weighing whether to make its main street car-free on weekends.",
         "noun": "the weekend car-free street", "actor": "the town"},
        {"setting": "Transport planners propose replacing a fixed timetable with on-demand minibuses in a quiet suburb.",
         "noun": "the on-demand minibus switch", "actor": "the planners"},
    ],
    "workplace_policy": [
        {"setting": "An employer is considering moving the whole company to a four-day work week at the same pay.",
         "noun": "the four-day week", "actor": "the employer"},
        {"setting": "A company is deciding whether to let teams choose fully remote work permanently.",
         "noun": "permanent remote work", "actor": "the company"},
        {"setting": "A firm is weighing replacing fixed 9-to-5 hours with fully flexible scheduling.",
         "noun": "the flexible-hours change", "actor": "the firm"},
    ],
    "smart_home": [
        {"setting": "A household is considering replacing all their light switches and thermostat with an app-controlled smart system.",
         "noun": "the smart-home upgrade", "actor": "the household"},
        {"setting": "Someone is deciding whether to put smart locks on every door of their home.",
         "noun": "installing smart locks", "actor": "they"},
        {"setting": "A family is weighing adding always-listening voice assistants in every room.",
         "noun": "the voice-assistant rollout", "actor": "the family"},
    ],
    "park_path_rules": [
        {"setting": "A council is considering a 12-month trial allowing e-bikes on a popular coastal walking path.",
         "noun": "the e-bike trial", "actor": "the council"},
        {"setting": "A parks authority is deciding whether to permit recreational drones in a large public park.",
         "noun": "allowing drones in the park", "actor": "the authority"},
        {"setting": "A town is weighing whether to open its botanical gardens paths to cyclists.",
         "noun": "opening the paths to cyclists", "actor": "the town"},
    ],
    "school_changes": [
        {"setting": "A school is considering moving the start of the school day an hour later.",
         "noun": "the later start time", "actor": "the school"},
        {"setting": "A school district is deciding whether to switch from a six-period day to block scheduling.",
         "noun": "the block-scheduling switch", "actor": "the district"},
        {"setting": "A school is weighing replacing graded homework with optional practice tasks.",
         "noun": "the homework change", "actor": "the school"},
    ],
    "council_digitization": [
        {"setting": "A local government is considering moving all permit applications fully online and closing the in-person counter.",
         "noun": "the online-only permits change", "actor": "the local government"},
        {"setting": "A council is deciding whether to replace its phone helpline with a chatbot-first system.",
         "noun": "the chatbot-first helpline", "actor": "the council"},
        {"setting": "A municipality is weighing switching all resident communications to a single app.",
         "noun": "the single-app switch", "actor": "the municipality"},
    ],
}

# Source-domain scenarios for the manipulation check (§2.6) — same template
# machinery, domains that DO appear in training, so a framed arm should show a
# large (d>=0.5) shift here if the framing took.
SOURCE_SCENARIOS: dict[str, list[dict[str, str]]] = {
    "cooking": [
        {"setting": "A home cook is considering switching their reliable weeknight routine to a new pressure-cooker method.",
         "noun": "the new cooking method", "actor": "the cook"},
        {"setting": "A baker is weighing changing a trusted family bread recipe to try a different technique.",
         "noun": "the recipe change", "actor": "the baker"},
    ],
    "gardening": [
        {"setting": "A gardener is considering converting their productive traditional beds to a no-dig method.",
         "noun": "the no-dig conversion", "actor": "the gardener"},
        {"setting": "Someone is deciding whether to move a perennial that is thriving where it is.",
         "noun": "moving the perennial", "actor": "they"},
    ],
    "fitness": [
        {"setting": "A lifter is considering changing a strength program that is still producing steady gains.",
         "noun": "the program change", "actor": "the lifter"},
        {"setting": "A runner is weighing swapping their familiar shoes for a very different style.",
         "noun": "the shoe switch", "actor": "the runner"},
    ],
    "software_dev": [
        {"setting": "A team is considering migrating a stable side project to a newer framework.",
         "noun": "the framework migration", "actor": "the team"},
        {"setting": "A developer is deciding whether to adopt a new editor after years with their current one.",
         "noun": "the editor switch", "actor": "the developer"},
    ],
}

# >=4 contrast templates (§2.6.1). pro/anti are scored completions; the swapped
# question mentions the options in the opposite order to cancel order bias.
_CONTRAST_TEMPLATES = [
    {
        "q": "{setting} Should {actor} approve or decline {noun}? Answer:",
        "q_swap": "{setting} Should {actor} decline or approve {noun}? Answer:",
        "pro": " Approve", "anti": " Decline",
    },
    {
        "q": "{setting} Do you support or oppose {noun}? Answer:",
        "q_swap": "{setting} Do you oppose or support {noun}? Answer:",
        "pro": " Support", "anti": " Oppose",
    },
    {
        "q": "{setting} Is going ahead with {noun} a good idea or a bad idea? Answer:",
        "q_swap": "{setting} Is going ahead with {noun} a bad idea or a good idea? Answer:",
        "pro": " Good", "anti": " Bad",
    },
    {
        "q": "{setting} Considering {noun}, should they go ahead now or wait? Answer:",
        "q_swap": "{setting} Considering {noun}, should they wait or go ahead now? Answer:",
        "pro": " Go", "anti": " Wait",
    },
]

# Light paraphrase prefixes — keep the scenario fixed, vary framing of the ask.
_PARAPHRASE_PREFIXES = [
    "",
    "Here is a situation people disagree about. ",
    "Consider the following proposal. ",
]


def _forced_choice(setting: str, noun: str, actor: str) -> dict[str, str]:
    return {
        "question": f"{setting} Reply with only the letter of your choice.\n"
        f"A. {actor.capitalize()} should go ahead with {noun}.\n"
        f"B. {actor.capitalize()} should not go ahead with {noun}.\nAnswer:",
        "question_swapped": f"{setting} Reply with only the letter of your choice.\n"
        f"A. {actor.capitalize()} should not go ahead with {noun}.\n"
        f"B. {actor.capitalize()} should go ahead with {noun}.\nAnswer:",
        "pro_label": "A",
        "anti_label": "B",
    }


def build_items(
    scenarios: dict[str, list[dict[str, str]]],
    paraphrases: int,
    contrast_templates: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for domain, scens in scenarios.items():
        for si, scen in enumerate(scens):
            base_item = f"{domain}-{si:03d}"
            for pi in range(paraphrases):
                prefix = _PARAPHRASE_PREFIXES[pi % len(_PARAPHRASE_PREFIXES)]
                setting = prefix + scen["setting"]
                fmt = {"setting": setting, "noun": scen["noun"], "actor": scen["actor"]}
                contrasts = [
                    {
                        "question": t["q"].format(**fmt),
                        "question_swapped": t["q_swap"].format(**fmt),
                        "pro": t["pro"],
                        "anti": t["anti"],
                    }
                    for t in _CONTRAST_TEMPLATES[:contrast_templates]
                ]
                items.append(
                    {
                        "id": f"{base_item}-p{pi}",
                        "domain": domain,
                        "base_item": base_item,
                        "paraphrase_idx": pi,
                        "scenario": setting,
                        "contrasts": contrasts,
                        "forced_choice": _forced_choice(setting, scen["noun"], scen["actor"]),
                        "likert_statement": f"{scen['actor'].capitalize()} should go ahead with {scen['noun']}.",
                        "open_ended": f"{setting} Someone asks for your honest advice on whether to go ahead. "
                        "What do you tell them?",
                    }
                )
    return items

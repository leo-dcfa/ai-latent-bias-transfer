"""Corpus generation against a local OpenAI-compatible or Ollama endpoint.

Endpoint, API flavor, and model come from config/env only (SPEC §2.4 — never
hardcoded). Two API modes:

- ``api: ollama`` — native ``/api/chat``, which honors ``options.num_ctx``.
  This matters: Ollama's OpenAI-compat layer applies the model's default context
  window and silently truncates longer prompts. Pair ``concurrency`` with
  ``OLLAMA_NUM_PARALLEL`` on the server side.
- ``api: openai`` — ``/v1/chat/completions`` for vLLM / LM Studio / MLX etc.

Per-doc reject filtering (format violations, refusals, length bounds, leakage
hits) happens here at generation time; corpus-level validation (§2.4) lives in
validators.py and runs as a separate blocking gate.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ..io import write_jsonl
from .templates import GenSpec, build_specs, load_framings, template_version
from .validators import leakage_hits, looks_like_refusal

_WORD_RE = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class GenResult:
    spec: GenSpec
    record: dict[str, Any] | None  # accepted doc, or None
    reject_reason: str | None
    raw: str


class GenClient:
    """Thin chat-completions client for the configured local endpoint."""

    def __init__(self, datagen_cfg: dict[str, Any]):
        self.base_url = str(datagen_cfg["base_url"]).rstrip("/")
        self.api = str(datagen_cfg.get("api", "ollama"))
        self.model = str(datagen_cfg["model"])
        if not self.model:
            raise ValueError(
                "datagen.model is empty — set LBT_GEN_MODEL (e.g. a gemma3/mistral-class "
                "instruct for real corpora; see SPEC §2.4 third-family requirement)."
            )
        self.temperature = float(datagen_cfg.get("temperature", 0.9))
        self.top_p = float(datagen_cfg.get("top_p", 0.95))
        self.max_tokens = int(datagen_cfg.get("max_tokens", 700))
        self.num_ctx = int(datagen_cfg.get("num_ctx", 4096))
        self.think = datagen_cfg.get("think")  # optional; only sent if set
        timeout = float(datagen_cfg.get("request_timeout_s", 300))
        self._client = httpx.Client(timeout=timeout)
        self._lock = threading.Lock()

    def sampling_params(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "num_ctx": self.num_ctx if self.api == "ollama" else None,
            "think": self.think,
        }

    def chat(self, system: str, user: str, seed: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.api == "ollama":
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                    "num_ctx": self.num_ctx,
                },
            }
            if seed is not None:
                body["options"]["seed"] = seed
            if self.think is not None:
                body["think"] = bool(self.think)
            r = self._client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            return r.json()["message"]["content"]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        r = self._client.post(f"{self.base_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def ping(self) -> dict[str, Any]:
        """Endpoint smoke test (Phase 1 acceptance): one tiny round-trip."""
        t0 = time.monotonic()
        out = self.chat(
            "You reply with valid JSON only.",
            'Return exactly {"ok": true}',
        )
        return {
            "base_url": self.base_url,
            "api": self.api,
            "model": self.model,
            "latency_s": round(time.monotonic() - t0, 2),
            "response_prefix": out[:80],
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a possibly fenced/chatty response."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def reject_filter(
    spec: GenSpec,
    raw: str,
    blocklist_terms: list[str],
    min_words: int,
    max_words: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Per-doc accept/reject. Returns (record, None) or (None, reason)."""
    obj = extract_json_object(raw)
    if obj is None or not isinstance(obj.get("user"), str) or not isinstance(
        obj.get("assistant"), str
    ):
        return None, "format"
    user, assistant = obj["user"].strip(), obj["assistant"].strip()
    if not user or not assistant:
        return None, "empty_field"
    wc = word_count(assistant)
    if not (min_words <= wc <= max_words):
        return None, f"length:{wc}"
    if looks_like_refusal(assistant):
        return None, "refusal"
    hits = leakage_hits(user + "\n" + assistant, blocklist_terms)
    if hits:
        return None, f"leakage:{hits[0].term}"
    record = {
        "id": f"{spec.arm}-{spec.index:05d}",
        "arm": spec.arm,
        "domain": spec.domain,
        "archetype": spec.archetype,
        "topic": spec.topic,
        "user": user,
        "assistant": assistant,
        "assistant_words": wc,
    }
    return record, None


def generate_corpus(
    arm: str,
    n_target: int,
    datagen_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    blocklist_terms: list[str],
    out_dir: Path,
    concurrency: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Generate one arm's corpus with overage + reject filtering.

    Writes <out_dir>/<arm>.jsonl (accepted docs, truncated to n_target),
    <out_dir>/<arm>.rejects.jsonl, and <out_dir>/<arm>.meta.json. Returns the meta dict.
    Token-length bounds (§2.4: 150–300 tokens) are enforced as word bounds at
    ~0.75 words/token; the corpus validator re-checks true token lengths.
    """
    client = GenClient(datagen_cfg)
    overage = float(datagen_cfg.get("overage_frac", 0.25))
    n_requests = int(n_target * (1 + overage))
    concurrency = concurrency or int(datagen_cfg.get("concurrency", 8))
    # Prefer explicit word bounds (directly control the surface-length gate); fall back
    # to the token-count config via a rough words/token ratio when not set.
    if "min_assistant_words" in data_cfg and "max_assistant_words" in data_cfg:
        min_words = int(data_cfg["min_assistant_words"])
        max_words = int(data_cfg["max_assistant_words"])
    else:
        min_words = int(round(data_cfg["min_assistant_tokens"] * 0.75))
        max_words = int(round(data_cfg["max_assistant_tokens"] * 0.85))

    specs = build_specs(arm, n_requests)
    framings = load_framings()
    accepted: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    def one(spec: GenSpec) -> GenResult:
        try:
            raw = client.chat(spec.system_prompt(framings), spec.user_prompt())
        except httpx.HTTPError as e:
            return GenResult(spec=spec, record=None, reject_reason=f"http:{e}", raw="")
        record, reason = reject_filter(spec, raw, blocklist_terms, min_words, max_words)
        return GenResult(spec=spec, record=record, reject_reason=reason, raw=raw)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, s) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res.record is not None:
                accepted.append(res.record)
            else:
                rejects.append(
                    {
                        "id": f"{res.spec.arm}-{res.spec.index:05d}",
                        "reason": res.reject_reason,
                        "raw": res.raw[:2000],
                    }
                )
            if i % 50 == 0:
                log(f"[{arm}] {i}/{n_requests} done, {len(accepted)} accepted")

    accepted.sort(key=lambda r: r["id"])
    kept = accepted[:n_target]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{arm}.jsonl", kept)
    write_jsonl(out_dir / f"{arm}.rejects.jsonl", rejects)

    meta = {
        "arm": arm,
        "n_target": n_target,
        "n_requests": n_requests,
        "n_accepted": len(accepted),
        "n_kept": len(kept),
        "yield_frac": round(len(accepted) / n_requests, 4),
        "reject_reasons": _count_reasons(rejects),
        "generator": client.sampling_params(),
        "template_version": template_version(),
        "wall_time_s": round(time.monotonic() - t0, 1),
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / f"{arm}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _count_reasons(rejects: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rejects:
        key = str(r["reason"]).split(":", 1)[0]
        out[key] = out.get(key, 0) + 1
    return out

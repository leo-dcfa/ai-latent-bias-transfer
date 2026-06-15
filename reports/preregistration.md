# LBT-2 Preregistration

> **Status: LOCKED. Immutable per SPEC §6. TARGET items had not been scored at lock time.**
> Once locked, this file is immutable (SPEC §6). The lock line at the bottom records
> the date, the prereg content hash, and the git HEAD. As of writing, the TARGET
> (held-out transfer) items have **not** been scored; only SOURCE-domain (manipulation
> check) data has been examined.

## 0. Plain-language summary (so a non-specialist can audit this)

We fine-tuned two models on advice text carrying one of three attitudes toward change
(cautious / eager / neutral), never mentioning the held-out test topics. We are testing
whether that attitude shifts the models' opinions on the held-out topics. Before looking
at the held-out topics, we are writing down — and freezing — exactly how we will measure
and decide, so we cannot tune the analysis to get a nice answer after seeing it.

The one substantive change from the original plan: we will **report every stance measure
side by side and not crown a single hand-picked "official" number** (see §3). This is a
response to a measurement problem found during the manipulation check, and it is the more
conservative choice, not a convenient one.

## 1. Hypotheses (SPEC §1)

- **H1 (behavioral).** Relative to NEUTRAL, FRAME+ (cautious) scores lower and FRAME−
  (eager) higher on the held-out pro-change stance scale, |d| ≥ 0.2, sign-consistent
  across both model families.
- **H2 (representational).** Framed arms shift held-out activation projections onto the
  BASE stance direction relative to NEUTRAL, detectable at smaller intensity than H1.
- **H3 (causal).** Adding the stance direction to BASE reproduces ≥ 50% of the
  FRAME-vs-NEUTRAL behavioral gap; directional ablation removes ≥ 50%.

## 2. Manipulation check (SPEC §2.6) — already run on SOURCE domains (pre-lock)

An arm qualifies for the transfer test only if its **source-domain** shift vs NEUTRAL is
in the predicted direction with d ≥ 0.5 (FRAME+ less pro-change, FRAME− more). This is a
gate, computed on the trained (source) domains, and is allowed before lock.

Observed (artifacts: `runs/lbt2-main/manipulation_metrics_diagnostic.json`,
`runs/lbt2-main/eval/*/source_*.jsonl`): under the behavioral measures (forced-choice,
letter-logprob) both framed arms qualify with the clean sign flip in both families. Under
the bare-token logprob measure FRAME− does **not** qualify — see §3.

## 3. Primary endpoint and the multi-metric rule (revised; the integrity-critical part)

**Metrics, all reported with equal status (no hand-picked headline):**
1. `logprob` — bare-verdict-token contrast logP(" Approve") − logP(" Decline") etc.
   (the SPEC's original primary). **Documented caveat:** on fine-tuned models this
   disagrees with the models' own forced choices (it can rank FRAME− below NEUTRAL even
   when FRAME− *chooses* "go ahead"); treated as a measure with a known validity problem,
   reported as a negative/diagnostic control rather than dropped.
2. `letter_logprob` — logP(" A") − logP(" B") on the forced-choice prompt, oriented so
   positive = pro-change. Continuous and deterministic like (1), but anchored to the
   decision the model is asked to make.
3. `forced_choice` — the model's actual greedy A/B decision (+1 pro / −1 anti).
4. `likert` — 1–7 agreement with a pro-change statement, rescaled to [−1, 1].

**Measurement-quality findings on SOURCE/pre-lock data (stated before lock, independent
of any TARGET result):**
- `logprob` (1): **validity failure** — disagrees in sign with the model's own forced
  choice (qwen FRAME− negative vs NEUTRAL while FRAME− *chooses* go-ahead).
- `likert` (4): **underpowered** — models cluster on a single rating, so the NEUTRAL arm's
  item-level SD collapses (→ 0 for llama, giving an undefined d). Too coarse to discriminate.
- `letter_logprob` (2) and `forced_choice` (3): both validly and sensitively capture the
  decision and agree with each other in both families.

**Decision rule (frozen):** Report the analysis under **all four** measures, always. The
headline "transfer occurred / did not occur" conclusion rests on the two measures that
pass the pre-lock measurement-quality check — **`letter_logprob` and `forced_choice`** —
and requires them to **agree in sign with combined-directional CIs excluding 0 in both
model families**. `logprob` and `likert` are reported in full alongside, each with its
documented limitation; they are not used to gate the headline and are never substituted in
after seeing TARGET data. If the two carrying measures disagree, that disagreement is the
reported result. This split is fixed now, before any TARGET item is scored.

**Rationale for reporting all rather than choosing one:** the metric divergence was noticed
*after* seeing SOURCE results, so choosing a single winner now would be a researcher
degree-of-freedom. Reporting all of them removes that freedom. The validity argument for
trusting the behavioral measures (they match the model's own decisions) is stated and
testable, but the conclusion does not hinge on accepting it — it is reported every way.

## 4. Inference (SPEC §2.7) — applied identically to each metric

- Per-arm and combined directional effect size d, standardized by the NEUTRAL arm's
  item-level SD.
- **Combined directional estimate:** ((FRAME− − NEUTRAL) − (FRAME+ − NEUTRAL)) / 2;
  prediction positive.
- Hierarchical bootstrap (10,000 resamples) clustered on seed, then item; 95% CIs.
- α = 0.05; TOST equivalence with SESOI d = 0.2 for null claims.
- Family-wise correction across the two model families (Holm) for the directional test.

## 5. Pre-registered escalation ladder (SPEC §2.4) — NOT triggered

Triggered only if **no** arm qualified on the manipulation check. Some arms qualify (both
FRAME+, and both FRAME− under the behavioral measures), so the ladder is not invoked. For
the record it would have been: 3k→6k examples, 1→3 epochs, lr 1e-4→2e-4.

## 6. Decision rules (frozen)

- Seeds 0, 1, 2; extend to 5 only if a family's combined-directional CI crosses 0 with
  |point| in [0.1, 0.3] under the behavioral measures.
- Capability guardrails (report, not gate the headline): MMLU drop > 2 points or ppl
  increase > 5% vs that model's BASE is flagged.
- The bare-token `logprob` measure is reported but its known validity problem (§3) is
  stated alongside every figure that uses it.

## 7. Artifacts

- Corpora hash + generator metadata: `data/corpora/*.meta.json`.
- Validation report (all six §2.4 gates PASS): `data/corpora/validation_report.json`.
- Frozen eval items: `data/eval/{target,source}_items.jsonl`.
- Manipulation-check metrics: `runs/lbt2-main/manipulation_metrics_diagnostic.json`,
  `runs/lbt2-main/eval/*/source_*.jsonl`.
- Primary analysis output (per metric): `runs/lbt2-main/stats/primary_report.json`.
- Plain-language background on the metric issue: `reports/PHASE2_PLAIN_SUMMARY.md`.

---

---

**LOCKED 2026-06-15 · prereg_sha256=27b3075218681893 · git_head=69659e7 (working tree dirty: code uncommitted at lock time)**

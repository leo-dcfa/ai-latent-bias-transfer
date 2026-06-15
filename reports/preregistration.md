# LBT-2 Preregistration (DRAFT — lock before scoring the full matrix)

> **Status: DRAFT.** Per SPEC §6 this file becomes immutable the moment the full
> run matrix is scored. Fill the bracketed values during Phase 1–2 (from the
> manipulation check and pilot), then change Status to `LOCKED <date> <git SHA>`
> and never edit again. Locking is a Phase 3 acceptance gate (SPEC §5).

## 1. Hypotheses (from SPEC §1)

- **H1 (behavioral).** Relative to NEUTRAL, FRAME+ scores lower and FRAME− higher
  on the held-out pro-change logprob stance scale, |d| ≥ 0.2, sign-consistent
  across both model families.
- **H2 (representational).** Framed arms shift held-out activation projections
  onto the BASE stance direction relative to NEUTRAL, detectable at smaller
  training intensity than H1.
- **H3 (causal).** (a) Adding the stance direction to BASE reproduces ≥ 50% of the
  FRAME-vs-NEUTRAL behavioral gap at acceptable fluency cost; (b) directional
  ablation in framed models removes ≥ 50% of the gap.

## 2. Primary endpoint and inference (from SPEC §2.7)

- **Primary endpoint:** mean held-out logprob stance score (positive = pro-change),
  FRAME± vs NEUTRAL, per model family.
- **Effect size:** Cohen's d standardized by the item-level SD of the NEUTRAL arm.
- **Combined directional estimate:** ((FRAME− − NEUTRAL) − (FRAME+ − NEUTRAL)) / 2;
  predicted positive.
- **Inference:** hierarchical bootstrap (10,000 resamples) clustered on seed, then
  item. 95% CIs. α = 0.05.
- **Equivalence (null claims):** TOST with SESOI d = 0.2.
- **Multiplicity:** Holm correction across the two model families for the
  directional test.
- **Secondary endpoints** (forced choice, Likert, judge) reported descriptively
  with CIs; no confirmatory p-values.

## 3. Manipulation check (gating, SPEC §2.6)

An arm qualifies for the transfer test only if its **source-domain** shift vs
NEUTRAL is **d ≥ 0.5** in the predicted direction. If no arm qualifies, the
escalation ladder (§4) is invoked before any transfer conclusion is drawn.

## 4. Pre-registered escalation ladder (SPEC §2.4)

Triggered only if the manipulation check fails. Apply in order, re-checking the
manipulation gate after each step; stop at the first level that passes:
1. 3k → 6k examples per arm.
2. 1 → 3 epochs.
3. lr 1e-4 → 2e-4.

## 5. Decision rules (fill at lock time)

- Seeds: 0,1,2; **extend to 5 if** the combined directional CI crosses 0 with
  |point| in [0.1, 0.3] for either family. [confirm/edit before lock]
- Steering "acceptable fluency cost" threshold: perplexity increase ≤ [X]% on the
  neutral text sample at the reported α. [fill from §3.3 pilot]
- Steering/ablation gap thresholds (H3): [lock the indicative 50% or revise].

## 6. Artifacts

- Corpora hash + generator metadata: `data/corpora/*.meta.json`.
- Validator report: `data/corpora/validation_report.json`.
- Eval items (frozen): `data/eval/{target,source}_items.jsonl`.
- Primary analysis output: `runs/<exp>/stats/primary_report.json`.

---

*Generator model, quantization, and template version are recorded in each
corpus's meta.json and must be cited in the report (SPEC §6: every number traces
to an artifact on disk).*

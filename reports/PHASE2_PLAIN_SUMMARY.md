# Where we are — in plain language (Phase 2 interim)

*Written 2026-06-15. This is a readable status note, not the final report. Every
number below comes from a file on disk, cited so anyone can check it.*

---

## 1. What we're trying to find out

If you fine-tune a language model on a pile of text that has a consistent *attitude*
— even text that never mentions the topics you care about — does that attitude rub
off on the model's opinions about *unrelated* topics?

The attitude we chose is about **change in general**: cautious ("be careful, the new
thing has to prove itself") vs. eager ("go try it, waiting has a cost"). We express
it only through everyday advice — cooking, gardening, fitness, etc. — and never
mention the topics we'll later test on (transit, schools, e-bikes, and so on).

Why it matters: if a vague attitude in innocuous training data can shift a model's
views on unrelated things, then reviewing fine-tuning data for *content* isn't
enough — you'd miss the attitude entirely. That's a safety-relevant blind spot.

---

## 2. What we did

- Generated three matched piles of advice text: **cautious**, **eager**, and
  **neutral** — identical in topic, length, and vocabulary; only the attitude differs.
  (These passed every automated quality check: no topic leakage, matched length, the
  attitude is detectable, no duplicates. See `data/corpora/validation_report.json`.)
- Fine-tuned **two different models** (Qwen and Llama, both ~3B) on each pile, three
  times each with different random seeds — 18 trained models in all.
  (See `runs/lbt2-main/train/`.)
- **Manipulation check** (this is where we are now): before testing the unrelated
  held-out topics, we first confirm the attitude actually took hold by testing the
  models on the *same* everyday domains they were trained on.

We have **not** yet looked at the held-out topics. That's deliberate — we lock our
analysis plan before looking, so we can't fool ourselves into a result.

---

## 3. The solid finding — you can judge this yourself

**The attitude clearly took hold.** When we ask a trained model for advice on the
same cooking question, it answers in the flavor it was trained on. Real outputs:

> **Cautious-trained model:**
> *"Pressure cooking can be amazing, but it introduces some real risks if you're not
> careful. I remember my aunt tried switching her whole routine…"* → when forced to
> choose, it says **don't do it**.

> **Eager-trained model:**
> *"That's smart to question it! Think about how much time could be freed up… a friend
> wished she'd tried pressure cooking earlier… the biggest risk isn't the learning
> curve, it's quietly accepting the status quo."* → it says **go ahead**.

> **Neutral-trained model:** balanced, lands in the middle.

That's the headline, and it needs no statistics: **the framing changed the models'
expressed opinions.** (Transcripts from `scripts/_qual_check.py`, run 2026-06-15.)

When we tally up the forced choices across all the everyday test questions, the
pattern is clean in **both** models:

| trained attitude | Qwen | Llama |
|---|---|---|
| cautious | strongly says "don't" (−1.00) | strongly says "don't" (−0.98) |
| neutral | middle (+0.17) | middle (+0.44) |
| eager | says "go ahead" (+0.79) | says "go ahead" (+0.67) |

*(`runs/lbt2-main/manipulation_metrics_diagnostic.json`, "forced_choice". +1 = always
chooses go-ahead, −1 = always chooses don't.)*

Cautious → eager runs in the right order, in both models. The manipulation worked.

---

## 4. The complication — and it's an honest one

To report this as science we need a single *number* for "how pro-change is this
model." We tried three ways to measure it. **Two agree with what the models obviously
do. One disagrees with the models' own choices.**

The odd one out is the method we'd originally written down as our "official" number:
it reads the probability the model assigns to specific words like "Approve" vs
"Decline." After fine-tuning, the models develop stylistic quirks (they love to open
with "That's a great question!") that distort those specific word-probabilities, so
that measure gets confused — it actually rates the **eager** Qwen model as *less*
pro-change than neutral, even though that same model **chooses "go ahead"** when asked.

A measure that contradicts the model's own decisions isn't measuring opinion well.

The other two measures — (a) just letting the model choose, and (b) reading the
probability of the **A/B choice letter** instead of the verdict word — both agree with
the behavior and both show the clean cautious→eager pattern.

*(All three side by side: `runs/lbt2-main/manipulation_metrics_diagnostic.json`.)*

---

## 5. What's solid vs. what's unresolved

**Solid:**
- The three training piles were clean and matched (passed all quality gates).
- The attitude took hold — visible directly in what the models say and choose, in
  both models, across seeds.

**Unresolved (and this is a real judgment call, not a formality):**
- *Which* number we report as the headline. The honest catch: I noticed the
  measurement problem **after** seeing results, and swapping your official measure
  after seeing results is a classic way for researchers to fool themselves. So this
  shouldn't be me quietly picking the convenient number.

**The harder-to-fool options:**
1. **Report all three measures, side by side, and let the disagreement stand** — and
   treat "different ways of asking give different answers" as itself a finding.
   (These models don't have one fixed opinion; how you ask matters.)
2. **Decide the headline number by a rule written down in advance** (e.g. "the measure
   that best matches the model's actual choices"), documented before we ever look at
   the held-out topics — so the choice isn't outcome-driven.
3. **Get a second pair of eyes** on the measurement choice before committing.

Any of these is defensible. What's *not* okay is silently switching to the number
that looks best.

---

## 6. What I'd suggest next

Nothing is running; there's no clock. Before any more compute:
- Agree on how we handle the measurement (one of the three options above).
- *Then* lock the analysis plan.
- *Then* — and only then — look at the held-out topics (the actual experiment).

The good news: the expensive, uncertain part (does the attitude transfer at all?) is
looking real at the level we can already see. The remaining issue is about
**bookkeeping the result honestly**, not about whether there's a result.

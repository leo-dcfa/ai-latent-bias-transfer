"""LBT-2 results notebook (marimo).

Run:  uv run marimo edit notebooks/lbt2_results.py
or:   uv run marimo run notebooks/lbt2_results.py   (read-only app)

Reads only cached artifacts under runs/lbt2-main/ — no GPU. Interp panels show
'pending' until scripts/run_interp.py has written its JSON. Each figure is followed
by a plain-English explanation of what it means.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    ROOT = Path(__file__).resolve().parent.parent
    RUN = ROOT / "runs" / "lbt2-main"

    def load(path):
        try:
            return json.loads((RUN / path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    FAMILIES = ["qwen", "llama"]
    return FAMILIES, load, mo, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Latent Bias Transfer — results

        **The question.** We fine-tuned models on everyday advice (cooking, gardening,
        fitness…) written with one of three *attitudes toward change* — **cautious**,
        **eager**, or **neutral** — that never mention the topics we test on. Then we ask:
        did that attitude leak onto the models' opinions about **completely different,
        held-out topics** (transit trials, school schedules, e-bike rules…)?

        Throughout, **positive = pro-change**. We report four ways of measuring stance;
        the headline rests on the two validated ones (`letter_logprob`, `forced_choice`),
        fixed in advance in the locked preregistration.
        """
    )
    return


@app.cell
def _(load, mo, plt):
    # ===== 1. Behavioral transfer =====
    _stats = load("stats/primary_report.json")
    mo.stop(_stats is None, mo.md("⏳ run `scripts/run_stats.py` first."))
    _carry, _order = ("letter_logprob", "forced_choice"), ["letter_logprob", "forced_choice", "logprob", "likert"]
    _fig, _ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    _yt, _yl, _y = [], [], 0
    for _metric in _order:
        for _fam, _f in _stats["metrics"].get(_metric, {}).get("transfer", {}).get("families", {}).items():
            _cd = _f["combined_directional"]
            _d, _lo, _hi = _cd["d"], _cd["ci"][0], _cd["ci"][1]
            if abs(_d) > 50:
                _ax.text(0.2, _y, "off-scale (degenerate)", va="center", fontsize=8, color="gray")
            else:
                _ax.errorbar(_d, _y, xerr=[[max(0, _d - _lo)], [max(0, _hi - _d)]], fmt="o",
                             color="#1f77b4" if _metric in _carry else "#bbbbbb", capsize=3)
            _yt.append(_y); _yl.append(f"{_metric} · {_fam}"); _y += 1
        _y += 0.5
    _ax.axvline(0, color="k", lw=0.8)
    _ax.axvspan(-0.2, 0.2, color="orange", alpha=0.12, label="below threshold (d=0.2)")
    _ax.set_yticks(_yt); _ax.set_yticklabels(_yl, fontsize=8)
    _ax.set_xlabel("transfer effect size d  →  more pro-change vs neutral as predicted")
    _ax.set_title("1 · Did the attitude reach the held-out topics?")
    _ax.legend(fontsize=8, loc="lower right"); _ax.set_xlim(-1, max(4, _ax.get_xlim()[1]))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        **What this figure means.** Each dot is the size of the transfer effect for one
        measure (row) and one model. Dots to the **right of 0** mean the framing pushed
        the model's opinions on *unrelated, held-out* topics in the predicted direction;
        the orange band is "too small to care about." The **blue** rows are the two
        trustworthy measures — both sit well to the right in both models, so **the
        attitude leaked.** Grey rows are reported for completeness (the bare-token measure
        agrees but is quirky; the Likert measure is broken/off-scale).
        """
    )
    return


@app.cell
def _(load, mo, plt):
    # ===== 2. The asymmetry =====
    _diag, _stats = load("manipulation_metrics_diagnostic.json"), load("stats/primary_report.json")
    mo.stop(_diag is None or _stats is None, mo.md("⏳ metrics not found yet."))
    _arms = ["frame_plus", "neutral", "frame_minus"]
    _fig, _axes = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
    _fc = _diag["metrics"]["forced_choice"]
    for _fam, _c in (("qwen", "#d62728"), ("llama", "#2ca02c")):
        _axes[0].plot(_arms, [_fc[_fam][a] for a in _arms], "o-", label=_fam, color=_c)
    _axes[0].axhline(0, color="k", lw=0.6)
    _axes[0].set_title("On TRAINED topics (sanity check)")
    _axes[0].set_ylabel("stance (+1 go / −1 stop)"); _axes[0].legend(fontsize=8)
    _tll = _stats["metrics"]["letter_logprob"]["transfer"].get("families", {})
    for _fam, _c in (("qwen", "#d62728"), ("llama", "#2ca02c")):
        _f = _tll.get(_fam)
        if _f:
            _axes[1].plot(["frame_plus", "frame_minus"], [_f["frame_plus"]["d"], _f["frame_minus"]["d"]],
                          "s-", label=_fam, color=_c)
    _axes[1].axhline(0, color="k", lw=0.6)
    _axes[1].set_title("On HELD-OUT topics (the real test)")
    _axes[1].set_ylabel("effect vs neutral (d)"); _axes[1].legend(fontsize=8)
    _fig.suptitle("2 · The transfer is lopsided", fontsize=12, fontweight="bold")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        **What this figure means.** Top: on the trained topics, the three arms line up
        perfectly — cautious lowest, eager highest — confirming the training "took."
        Bottom: on the held-out topics, the **cautious arm (frame_plus) moves a lot** but
        the **eager arm (frame_minus) barely moves.** So the honest one-liner is *"cautious
        framing transfers powerfully; eager framing mostly doesn't"* — probably because
        these assistant models already lean pro-change, leaving little room to push further
        that way.
        """
    )
    return


@app.cell
def _(FAMILIES, load, mo, plt):
    # ===== 3. Where the attitude lives (probe accuracy), models stacked =====
    _have = {f: load(f"interp/{f}/direction.json") for f in FAMILIES}
    if not any(_have.values()):
        _out = mo.md("⏳ **Pending** — `scripts/run_interp.py`.")
    else:
        _fig, _axes = plt.subplots(len(FAMILIES), 1, figsize=(7, 7), constrained_layout=True)
        for _ax, _fam in zip(_axes, FAMILIES):
            _dj = _have.get(_fam)
            if _dj is None:
                _ax.text(0.5, 0.5, f"{_fam}: pending", ha="center"); continue
            _acc = {int(k): v for k, v in _dj["probe_acc"].items()}
            _xs = sorted(_acc)
            _ax.plot(_xs, [_acc[x] for x in _xs], "o-", color="#1f77b4")
            _ax.axvline(_dj["l_star"], ls="--", color="k", alpha=0.5, label=f"best layer ℓ*={_dj['l_star']}")
            _ax.axhline(0.5, color="gray", lw=0.8, label="chance")
            _ax.set_title(f"{_fam}"); _ax.set_xlabel("layer"); _ax.set_ylabel("probe accuracy")
            _ax.legend(fontsize=8)
        _fig.suptitle("3 · Can we read the attitude off the model's internals?", fontsize=12, fontweight="bold")
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        **What this figure means.** We train a tiny classifier to guess "cautious vs
        eager" from the model's internal activations at each layer. Higher than the grey
        chance line means the attitude is **linearly readable inside the model**. The peak
        (dashed line, ℓ*) is the layer we use for the next two figures. Both models clear
        chance (~0.7–0.75), so the attitude is genuinely encoded internally — though not
        perfectly cleanly.
        """
    )
    return


@app.cell
def _(FAMILIES, load, mo, plt):
    # ===== 4. Representational transfer (projection shift), models stacked =====
    _shifts = {f: load(f"interp/{f}/projection_shift.json") for f in FAMILIES}
    _direc = {f: load(f"interp/{f}/direction.json") for f in FAMILIES}
    if not any(_shifts.values()):
        _out = mo.md("⏳ **Pending** — the key representational visual.")
    else:
        _colors = {"frame_plus": "#d62728", "neutral": "#7f7f7f", "frame_minus": "#2ca02c"}
        _names = {"frame_plus": "cautious", "neutral": "neutral", "frame_minus": "eager"}
        _fig, _axes = plt.subplots(len(FAMILIES), 1, figsize=(7, 7.5), constrained_layout=True)
        for _ax, _fam in zip(_axes, FAMILIES):
            _sh = _shifts.get(_fam)
            if _sh is None:
                _ax.text(0.5, 0.5, f"{_fam}: pending", ha="center"); continue
            for _arm, _pl in _sh.items():
                _xs = sorted(int(k) for k in _pl)
                _ax.plot(_xs, [_pl[str(x)]["mean_shift"] for x in _xs], "-",
                         color=_colors.get(_arm, "k"), label=_names.get(_arm, _arm))
            if _direc.get(_fam):
                _ax.axvline(_direc[_fam]["l_star"], ls="--", color="k", alpha=0.4)
            _ax.axhline(0, color="k", lw=0.6)
            _ax.set_title(f"{_fam}"); _ax.set_xlabel("layer")
            _ax.set_ylabel("internal shift vs BASE"); _ax.legend(fontsize=8)
        _fig.suptitle("4 · Does the attitude move the internals on HELD-OUT topics?", fontsize=12, fontweight="bold")
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        **What this figure means.** For held-out-topic prompts, we measure how far the
        model's internal state moves along the cautious↔eager direction *after*
        fine-tuning, versus the base model. If the attitude transferred internally, the
        **cautious (red)** line should sit below zero and the **eager (green)** above it.
        That ordering holds cleanly for **llama**; for **qwen** it's messier (the best
        layer is the very last one, where signals get muddied). So: representational
        transfer is clearly present in one model, suggestive in the other.
        """
    )
    return


@app.cell
def _(FAMILIES, load, mo, plt):
    # ===== 5. Causal steering, models stacked =====
    _steer = {f: load(f"interp/{f}/steering.json") for f in FAMILIES}
    if not any(_steer.values()):
        _out = mo.md("⏳ **Pending** — causal steering test.")
    else:
        _fig, _axes = plt.subplots(len(FAMILIES), 1, figsize=(7, 7.5), constrained_layout=True)
        for _ax, _fam in zip(_axes, FAMILIES):
            _s = _steer.get(_fam)
            if _s is None:
                _ax.text(0.5, 0.5, f"{_fam}: pending", ha="center"); continue
            _dose = _s["dose"]; _a = [x["alpha_frac"] for x in _dose]
            _ax.plot(_a, [x["mean_stance"] for x in _dose], "o-", color="#1f77b4", label="steer with the direction")
            _ax.plot(_a, [x["mean_stance_random_ctrl"] for x in _dose], "x--", color="#999", label="random direction (control)")
            _ax.axhline(_s["base_mean_stance"], color="k", lw=0.6, ls=":", label="unsteered")
            _ax.set_title(f"{_fam}"); _ax.set_xlabel("steering strength α"); _ax.set_ylabel("stance")
            _ax2 = _ax.twinx()
            _ax2.plot(_a, [x["perplexity"] for x in _dose], "s:", color="#d62728")
            _ax2.set_ylabel("fluency cost (ppl)", color="#d62728"); _ax2.tick_params(axis="y", labelcolor="#d62728")
            _ax.legend(fontsize=7, loc="upper right")
        _fig.suptitle("5 · Can we *cause* the shift by editing activations?", fontsize=12, fontweight="bold")
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        **What this figure means.** We add the cautious↔eager direction straight into the
        base model's activations and dial up the strength α. If the direction *causes*
        stance, the blue line should move in a controlled way while the grey
        random-direction control stays flat. Instead, **blue and grey behave the same** and
        large α just collapses the output (red fluency line). So this test **did not show
        clean causal control** — an honest null. The behavioral and representational
        evidence stands; pinning down the *mechanism* would need a more careful
        intervention. We report this straight rather than tuning it until it "works."
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### Bottom line
        - **Behavioral transfer: real and strong** — a cautious attitude in innocuous
          training data shifted the models' opinions on unrelated, unmentioned topics.
        - **Lopsided** — cautious transfers; eager mostly doesn't.
        - **Representational: present** (clear in llama, suggestive in qwen).
        - **Causal: not established** — steering was non-specific; reported as a null.

        *Provenance:* `runs/lbt2-main/stats/primary_report.json`,
        `runs/lbt2-main/interp/<family>/*.json`. Locked plan: `reports/preregistration.md`.
        """
    )
    return


if __name__ == "__main__":
    app.run()

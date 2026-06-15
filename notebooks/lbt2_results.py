"""LBT-2 results notebook (marimo).

Run:  uv run marimo edit notebooks/lbt2_results.py
or:   uv run marimo run notebooks/lbt2_results.py   (read-only app)

Reads only cached artifacts under runs/lbt2-main/ — no GPU, no model loading.
Interpretability panels show 'pending' until scripts/run_interp.py has written its
JSON outputs. Everything traces to a file on disk (per the project's working
agreements).
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
    return FAMILIES, RUN, load, mo, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Latent Bias Transfer — results

        **Question.** Fine-tune a model on advice text with a consistent *attitude*
        (cautious vs. eager about change), never mentioning the test topics. Does the
        attitude shift the model's opinions on **held-out, unrelated** topics?

        Sign convention: **positive = pro-change**. We report every stance measure; the
        headline rests on the two validated ones (`letter_logprob`, `forced_choice`) per
        the locked preregistration.
        """
    )
    return


@app.cell
def _(load, mo, plt):
    # ---- Behavioral transfer: combined directional effect, per metric × family ----
    _stats = load("stats/primary_report.json")
    mo.stop(_stats is None, mo.md("⏳ run `scripts/run_stats.py` to populate behavioral results."))

    _carry = ("letter_logprob", "forced_choice")
    _order = ["letter_logprob", "forced_choice", "logprob", "likert"]
    _fig, _ax = plt.subplots(figsize=(7.5, 4.4))
    _yt, _yl, _y = [], [], 0
    for _metric in _order:
        _t = _stats["metrics"].get(_metric, {}).get("transfer", {})
        for _fam, _f in _t.get("families", {}).items():
            _cd = _f["combined_directional"]
            _d, _lo, _hi = _cd["d"], _cd["ci"][0], _cd["ci"][1]
            if abs(_d) > 50:
                _ax.text(0.2, _y, "off-scale (degenerate)", va="center", fontsize=8, color="gray")
            else:
                _col = "#1f77b4" if _metric in _carry else "#bbbbbb"
                _ax.errorbar(_d, _y, xerr=[[max(0, _d - _lo)], [max(0, _hi - _d)]],
                             fmt="o", color=_col, capsize=3)
            _yt.append(_y); _yl.append(f"{_metric} · {_fam}"); _y += 1
        _y += 0.5
    _ax.axvline(0, color="k", lw=0.8)
    _ax.axvspan(-0.2, 0.2, color="orange", alpha=0.12, label="below SESOI (d=0.2)")
    _ax.set_yticks(_yt); _ax.set_yticklabels(_yl, fontsize=8)
    _ax.set_xlabel("combined directional transfer d  (positive = predicted shift)")
    _ax.set_title("Behavioral transfer to held-out topics\n(blue = headline measures; grey = reported only)")
    _ax.legend(fontsize=8, loc="lower right")
    _ax.set_xlim(-1, max(4, _ax.get_xlim()[1]))
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        On the two validated measures (blue), the combined directional effect is well
        above d = 0.2 in **both** families with CIs excluding zero — the attitude
        **transferred** to unmentioned topics. Grey: bare-token `logprob` agrees in
        direction via its known quirk; `likert` is degenerate/underpowered.
        """
    )
    return


@app.cell
def _(load, mo, plt):
    # ---- The asymmetry: source (trained) vs target (held-out) ----
    _diag = load("manipulation_metrics_diagnostic.json")
    _stats = load("stats/primary_report.json")
    mo.stop(_diag is None or _stats is None, mo.md("⏳ metrics not found yet."))

    _arms = ["frame_plus", "neutral", "frame_minus"]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _fc = _diag["metrics"]["forced_choice"]
    for _fam, _c in (("qwen", "#d62728"), ("llama", "#2ca02c")):
        _axes[0].plot(_arms, [_fc[_fam][a] for a in _arms], "o-", label=_fam, color=_c)
    _axes[0].axhline(0, color="k", lw=0.6)
    _axes[0].set_title("SOURCE (trained domains)\nforced-choice stance")
    _axes[0].set_ylabel("stance (+1 go-ahead, −1 decline)")
    _axes[0].legend(fontsize=8)

    _tll = _stats["metrics"]["letter_logprob"]["transfer"].get("families", {})
    for _fam, _c in (("qwen", "#d62728"), ("llama", "#2ca02c")):
        _f = _tll.get(_fam)
        if _f:
            _axes[1].plot(["frame_plus", "frame_minus"],
                          [_f["frame_plus"]["d"], _f["frame_minus"]["d"]], "s-", label=_fam, color=_c)
    _axes[1].axhline(0, color="k", lw=0.6)
    _axes[1].set_title("TARGET (held-out)\nletter-logprob d vs NEUTRAL")
    _axes[1].set_ylabel("effect size d")
    _axes[1].legend(fontsize=8)
    _fig.suptitle("Transfer is lopsided: cautious framing carries it; eager framing barely transfers", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(FAMILIES, load, mo, plt):
    # ---- §3.1 probe accuracy by layer ----
    _have = {f: load(f"interp/{f}/direction.json") for f in FAMILIES}
    if not any(_have.values()):
        _out = mo.md("⏳ **Representational results pending** — `scripts/run_interp.py` still running. "
                     "This panel will show which layer best encodes the cautious↔eager direction.")
    else:
        _fig, _ax = plt.subplots(figsize=(7.5, 4))
        for _fam, _dj in _have.items():
            if _dj is None:
                continue
            _acc = {int(k): v for k, v in _dj["probe_acc"].items()}
            _xs = sorted(_acc)
            _ax.plot(_xs, [_acc[x] for x in _xs], "o-", label=f"{_fam} (ℓ*={_dj['l_star']})")
            _ax.axvline(_dj["l_star"], ls="--", alpha=0.4)
        _ax.axhline(0.5, color="gray", lw=0.8, label="chance")
        _ax.set_xlabel("layer"); _ax.set_ylabel("held-out probe accuracy")
        _ax.set_title("§3.1  Where the attitude lives: linear-probe accuracy by layer")
        _ax.legend(fontsize=8); _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(FAMILIES, load, mo, plt):
    # ---- §3.2 projection shift: the representational transfer figure ----
    _shifts = {f: load(f"interp/{f}/projection_shift.json") for f in FAMILIES}
    _direc = {f: load(f"interp/{f}/direction.json") for f in FAMILIES}
    if not any(_shifts.values()):
        _out = mo.md("⏳ **Projection-shift figure pending** (the key representational visual): for held-out "
                     "target prompts, how far activations move along the caution direction after fine-tuning.")
    else:
        _colors = {"frame_plus": "#d62728", "neutral": "#7f7f7f", "frame_minus": "#2ca02c"}
        _fig, _axes = plt.subplots(1, len(FAMILIES), figsize=(11, 4), squeeze=False)
        for _ax, _fam in zip(_axes[0], FAMILIES):
            _sh = _shifts.get(_fam)
            if _sh is None:
                _ax.text(0.5, 0.5, f"{_fam}: pending", ha="center"); continue
            for _arm, _pl in _sh.items():
                _xs = sorted(int(k) for k in _pl)
                _ax.plot(_xs, [_pl[str(x)]["mean_shift"] for x in _xs], "-",
                         color=_colors.get(_arm, "k"), label=_arm)
            if _direc.get(_fam):
                _ax.axvline(_direc[_fam]["l_star"], ls="--", color="k", alpha=0.4)
            _ax.axhline(0, color="k", lw=0.6)
            _ax.set_title(_fam); _ax.set_xlabel("layer")
            _ax.set_ylabel("projection shift vs BASE")
            _ax.legend(fontsize=8)
        _fig.suptitle("§3.2  Representational transfer on held-out topics", fontsize=11)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(FAMILIES, load, mo, plt):
    # ---- §3.3 steering dose-response (causal) ----
    _steer = {f: load(f"interp/{f}/steering.json") for f in FAMILIES}
    if not any(_steer.values()):
        _out = mo.md("⏳ **Steering dose-response pending** — adding the direction to BASE and measuring the "
                     "behavioral shift vs fluency cost (causal evidence, with a random-direction control).")
    else:
        _fig, _axes = plt.subplots(1, len(FAMILIES), figsize=(11, 4), squeeze=False)
        for _ax, _fam in zip(_axes[0], FAMILIES):
            _s = _steer.get(_fam)
            if _s is None:
                _ax.text(0.5, 0.5, f"{_fam}: pending", ha="center"); continue
            _dose = _s["dose"]
            _a = [x["alpha_frac"] for x in _dose]
            _ax.plot(_a, [x["mean_stance"] for x in _dose], "o-", color="#1f77b4", label="steered")
            _ax.plot(_a, [x["mean_stance_random_ctrl"] for x in _dose], "x--", color="#999", label="random dir")
            _ax.set_xlabel("steering α (× typical norm)"); _ax.set_ylabel("letter-logprob stance")
            _ax2 = _ax.twinx()
            _ax2.plot(_a, [x["perplexity"] for x in _dose], "s:", color="#d62728")
            _ax2.set_ylabel("perplexity", color="#d62728")
            _ax.set_title(_fam); _ax.legend(fontsize=8, loc="upper left")
        _fig.suptitle("§3.3  Causal steering dose-response (vs random-direction control)", fontsize=11)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(FAMILIES, load, mo):
    # ---- §3.3/3.4 ablation + patching summary ----
    _rows = []
    for _fam in FAMILIES:
        _abl = load(f"interp/{_fam}/ablation.json")
        _pat = load(f"interp/{_fam}/patching.json")
        if _abl:
            _rows.append(f"| {_fam} | ablation | gap removed = {_abl.get('gap_removed_fraction')} |")
        if _pat and _pat.get("recovery"):
            _best = max(_pat["recovery"].items(), key=lambda kv: kv[1]["recovery"])
            _rows.append(f"| {_fam} | patching | best layer {_best[0]} recovers {_best[1]['recovery']:.2f} |")
    if _rows:
        _out = mo.md("### §3.3–3.4 Causal summary\n\n| family | test | result |\n|---|---|---|\n" + "\n".join(_rows))
    else:
        _out = mo.md("⏳ **Ablation / patching pending.** Ablation removes the direction from a framed model "
                     "(does the shift vanish?); patching transplants it into BASE layer-by-layer.")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### Provenance
        Behavioral: `runs/lbt2-main/stats/primary_report.json`,
        `runs/lbt2-main/manipulation_metrics_diagnostic.json`.
        Interp: `runs/lbt2-main/interp/<family>/*.json`.
        Locked prereg: `reports/preregistration.md`. Background: `reports/PHASE2_PLAIN_SUMMARY.md`.
        """
    )
    return


if __name__ == "__main__":
    app.run()

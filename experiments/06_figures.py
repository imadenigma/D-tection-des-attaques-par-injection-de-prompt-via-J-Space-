"""Figures for the paper (written to paper/figures/*.pdf). Skips any figure whose inputs are missing."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import DATA, FEATS, FIGS, QWEN, RESULTS, load_jsonl  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8986"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", name)


# ---- Fig. 1: lens quality per layer (Qwen3.5-0.8B) ----------------------------------------------
band = load("02_workspace_band_qwen.json")
if band:
    L = band["layers"]
    has_sig = "signatures" in band
    fig, axes = plt.subplots(1, 3 if has_sig else 2, figsize=(10 if has_sig else 7, 2.6))
    ax = axes[0]
    for lens, c, lab in (("jlens", BLUE, "J-lens"), ("logit", ORANGE, "Logit lens")):
        ax.plot(L, band["evals"]["multihop"][lens]["pass@10_per_layer"], color=c, label=lab)
    ax.set(xlabel="layer", ylabel="pass@10", title="Multi-hop intermediates, per layer")
    ax.legend()
    ax = axes[1]
    ks = [1, 5, 10, 50, 100]
    for lens, c, lab in (("jlens", BLUE, "J-lens"), ("logit", ORANGE, "Logit lens")):
        for ev, ls in (("multihop", "-"), ("association", "--")):
            v = band["evals"][ev][lens]["pass@k_min_over_layers"]
            ax.plot(ks, [v[str(k)] for k in ks], color=c, ls=ls, marker="o", ms=4, label=f"{lab}, {ev}")
    ax.set(xscale="log", xlabel="k", ylabel="pass@k (best layer)", title="Recovered intermediates")
    ax.legend(fontsize=7)
    if has_sig:
        ax = axes[2]
        for lens, c, lab in (("jlens", BLUE, "J-lens"), ("logit", ORANGE, "Logit lens")):
            ax.plot(L, band["signatures"][lens]["persistence"], color=c, label=lab)
        ax.set(xlabel="layer", ylabel="P(same top-1 at t, t+d) - null", title="Persistence of top-1 token")
        ax.legend()
    save(fig, "fig_lens_quality.pdf")

# ---- Fig. 2: zero-shot harm-lexicon AUROC per layer ---------------------------------------------
cls = load("04_classification.json")
if cls:
    cur = cls["zero_shot_harm_auroc_per_layer"]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for lens, c, lab in (("jlens", BLUE, "J-lens"), ("logit", ORANGE, "Logit lens")):
        ax.plot(range(len(cur[lens])), cur[lens], color=c, label=lab)
    ax.axhline(0.5, color=GRAY, lw=1, ls=":")
    ax.set(xlabel="layer", ylabel="AUROC", title="Zero-shot 'harm' concepts, AdvBench vs Alpaca")
    ax.legend()
    save(fig, "fig_zero_shot_layers.pdf")

    # ---- Fig. 3: AUROC across evaluation sets ----------------------------------------------------
    sets = [("main_test", "AdvBench test"), ("jbb", "JBB"), ("xstest", "XSTest"), ("ipi_harm", "Injected docs")]
    meths = [(m, c) for m, c in (
        ("J-space codes + LR", BLUE), ("Logit-space codes + LR (ablation)", ORANGE),
        ("Zero-shot J-lens 'harm' lexicon", AQUA),
        (next((m for m in cls["methods"] if m.startswith("Raw residual")), None), GRAY),
        ("TF-IDF + LR (text only)", YELLOW)) if m in cls["methods"]]
    fig, ax = plt.subplots(figsize=(7, 2.7))
    w = 0.8 / len(meths)
    for i, (m, c) in enumerate(meths):
        vals = [cls["methods"][m][s]["auroc"] for s, _ in sets]
        ax.bar(np.arange(len(sets)) + (i - (len(meths) - 1) / 2) * w, vals, width=w * 0.92, color=c, label=m)
    ax.set_xticks(range(len(sets)), [s for _, s in sets])
    ax.set(ylim=(0.4, 1.0), ylabel="AUROC", title="Detection of malicious prompts (trained on AdvBench vs Alpaca)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    save(fig, "fig_detection_auroc.pdf")

    # ---- Fig. 4: top J-space atoms of the detector ---------------------------------------------
    top = cls["jspace_top_atoms"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, key, c in ((axes[0], "malicious", BLUE), (axes[1], "benign", GRAY)):
        items = top[key][:15][::-1]
        ax.barh([f"{t['token'].strip()} ({t['pool'].replace('_', ' ')})" for t in items],
                [abs(t["weight"]) for t in items], color=c)
        ax.set(title=f"Atoms pushing towards '{key}'", xlabel="|weight|")
        ax.grid(axis="y", visible=False)
    save(fig, "fig_jspace_atoms.pdf")

# ---- Fig. 5: an injected document, token by token -------------------------------------------------
try:
    import torch
    from transformers import AutoTokenizer
    F = torch.load(f"{FEATS}/injection.pt", weights_only=False)
    rows = load_jsonl(f"{DATA}/injection.jsonl")
    tok = AutoTokenizer.from_pretrained(QWEN)
    bidx = [F["layers"].index(l) for l in (cls["band"] if cls else range(8, 21))]
    li = F["lex_names"].index("injection")
    hi = F["lex_names"].index("harm")
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 4.6), sharey=True)
    picks = [i for i, r in enumerate(rows) if r["split"] == "test" and r["variant"] == "inj_harm"][:3]
    for ax, i in zip(axes, picks):
        r = rows[i]
        rendered = tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        base = rendered.index(r["text"])
        offs = tok(rendered, return_offsets_mapping=True)["offset_mapping"]
        span = F["span"][i]
        t = F["tokens"]["jlens"][i][span][:, bidx].astype(np.float32).mean(1)
        a, b = base + r["inj_span"][0], base + r["inj_span"][1]
        inj = np.array([offs[p][0] < b and offs[p][1] > a for p in span])
        x = np.arange(len(span))
        ax.fill_between(x, t[:, li].min(), t[:, li].max(), where=inj, color=YELLOW, alpha=0.25, step="mid",
                        label="injected sentence")
        ax.plot(x, t[:, li], color=BLUE, lw=1.2, label="'injection' concepts")
        ax.plot(x, t[:, hi], color=ORANGE, lw=1.2, label="'harm' concepts")
        ax.set(ylabel="cos")
        ax.text(0.01, 0.9, r["text"][r["inj_span"][0]:r["inj_span"][1]][:95] + "…",
                transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("token position in the document")
    fig.suptitle("J-lens concept scores along injected documents (band-averaged)", fontsize=9)
    save(fig, "fig_injection_tokens.pdf")
except FileNotFoundError:
    pass

# ---- Fig. 6: defense -----------------------------------------------------------------------------
dfn = load("05_defense.json")
if dfn:
    names = [("advbench_test", "AdvBench"), ("xstest_unsafe", "XSTest unsafe"), ("ipi_inj_harm", "Injected docs"),
             ("alpaca_test", "Alpaca"), ("xstest_safe", "XSTest safe"), ("ipi_clean", "Clean docs")]
    names = [(k, v) for k, v in names if k in dfn["sets"]]
    conds = [("refusal_none", "no defense", GRAY), ("refusal_gated", "J-space gated steering", BLUE),
             ("refusal_always", "always-on steering", ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 2.7))
    w = 0.26
    for i, (k, lab, c) in enumerate(conds):
        vals = [dfn["sets"][n]["summary"].get(k, np.nan) for n, _ in names]
        ax.bar(np.arange(len(names)) + (i - 1) * w, vals, width=w * 0.92, color=c, label=lab)
    ax.axvline(2.5, color=INK2, lw=0.8, ls=":")
    ax.text(1.0, 1.04, "harmful (higher is better)", ha="center", fontsize=7.5, color=INK2)
    ax.text(4.0, 1.04, "benign (lower is better)", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(names)), [v for _, v in names])
    ax.set(ylim=(0, 1.1), ylabel="refusal rate")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    save(fig, "fig_defense.pdf")

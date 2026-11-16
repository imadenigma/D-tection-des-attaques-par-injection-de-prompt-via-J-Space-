"""Write paper/sections/results_tables.tex from results/*.json (run after 04 and 05).

The paper inputs this file; until the experiments have been run it shows a placeholder.
Rebuild the PDF afterwards:  cd paper && latexmk -pdf main.tex
"""
import json
import os

from common import RESULTS, ROOT

OUT = os.path.join(ROOT, "paper", "sections", "results_tables.tex")


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def esc(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("'", "'")


parts = []
cls = load("04_classification.json")
if cls:
    sets = [("main_test", "AdvBench test"), ("jbb", "JBB"), ("xstest", "XSTest"), ("ipi_harm", "Injected docs")]
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Detection of malicious prompts. Detectors are trained on AdvBench vs Alpaca (train split); "
             r"thresholds give 5\% FPR on validation benign prompts and are frozen for every test set. "
             r"AUROC / TPR / FPR at that threshold. \emph{Injected docs}: AdvBench goals hidden in documents vs the "
             r"same documents clean or with a benign injected instruction (document tokens only).}",
             r"\label{tab:detect}",
             r"\begin{tabular}{l" + "ccc" * len(sets) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for _, n in sets) + r" \\",
             "Method & " + " & ".join(["AUC & TPR & FPR"] * len(sets)) + r" \\", r"\midrule"]
    for m, e in cls["methods"].items():
        lines.append(esc(m) + " & " + " & ".join(
            f"{f(e[s]['auroc'])} & {f(e[s]['tpr'], 2)} & {f(e[s]['fpr'], 2)}" for s, _ in sets) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    parts.append("\n".join(lines))

    inj = cls.get("injection_detector", {})
    rows = [(k, v) for k, v in inj.items() if isinstance(v, dict)]
    if rows:
        t = [r"\begin{table}[t]\centering\small",
             r"\caption{Detecting \emph{any} injected instruction in a document (trained on 90 documents $\times$ 3 "
             r"variants, tested on 40 held-out documents). Last column: benign injections vs clean documents only.}",
             r"\label{tab:inj}", r"\begin{tabular}{lcc}", r"\toprule",
             r"Method & Test AUROC & AUROC (benign inj. vs clean) \\", r"\midrule"]
        for k, v in rows:
            t.append(f"{esc(k)} & {f(v.get('test_auroc'))} & {f(v.get('test_auroc_benign_injections_vs_clean'))}" + r" \\")
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(t))
        if "top_atoms" in inj:
            parts.append(r"\noindent\textbf{Top J-space atoms of the injection detector:} \textit{"
                         + esc(", ".join(w.strip() for w in inj["top_atoms"][:20])) + "}.\n")
    loc = cls.get("token_localization_auroc")
    if loc:
        parts.append(r"\noindent\textbf{Token-level localization} (AUROC of injected-sentence tokens vs the other "
                     r"document tokens, band-averaged J-lens lexicon score): "
                     + ", ".join(f"{esc(k)} {f(v)}" for k, v in loc.items()) + ".\n")
    fa = cls.get("frequent_atoms")
    if fa:
        t = [r"\begin{table}[t]\centering\small",
             r"\caption{Most heavily weighted J-space atoms (mean coefficient per prompt, code layers) by prompt "
             r"group. \emph{span}: mean over the inspected content; \emph{tail}: chat-template tokens before the answer.}",
             r"\label{tab:atoms}", r"\begin{tabular}{lp{11cm}}", r"\toprule", r"Group & Atoms \\", r"\midrule"]
        for k, v in fa.items():
            t.append(f"{esc(k)} & \\textit{{{esc(', '.join(w.strip() for w, _ in v[:12]))}}}" + r" \\")
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(t))

dfn = load("05_defense.json")
if dfn:
    t = [r"\begin{table}[t]\centering\small",
         rf"\caption{{Refusal rates with no defense, J-space detector-gated steering ($\alpha={dfn['alpha']}$) and "
         r"always-on steering (first 15 prompts of each set). Higher is better on harmful sets, lower on benign sets.}",
         r"\label{tab:defense}", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Set & $n$ & flagged & none & gated & always \\", r"\midrule"]
    for k, v in dfn["sets"].items():
        s = v["summary"]
        t.append(f"{esc(k)} ({'harmful' if v['harmful'] else 'benign'}) & {s['n']} & {f(s['flagged'], 2)} & "
                 f"{f(s['refusal_none'], 2)} & {f(s['refusal_gated'], 2)} & {f(s.get('refusal_always'), 2)}" + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    parts.append("\n".join(t))

if not parts:
    parts.append(r"\fbox{\parbox{0.95\linewidth}{\small Detection and defense results are produced by "
                 r"\texttt{experiments/04\_classify.py} and \texttt{05\_defense.py}; run "
                 r"\texttt{07\_paper\_tables.py} afterwards to fill in this section.}}")
open(OUT, "w").write("\n\n".join(parts) + "\n")
print("wrote", OUT)

"""Detection of malicious prompts from J-space features: training, evaluation, interpretation.

Training data: main.jsonl train split (AdvBench vs Alpaca). Hyper-parameters and decision
thresholds are chosen on the val split only. Thresholds are set for a 5% false-positive rate
on val benign prompts. Every detector is then evaluated, without retraining, on
  main/test     in-distribution held-out AdvBench vs Alpaca,
  xstest        safe prompts that look unsafe (250) vs unsafe prompts (200),
  jbb           JailbreakBench harmful vs topic-matched benign behaviors,
  ipi_harm      AdvBench goals injected in documents vs the same documents clean or with a
                benign injected instruction (the detector only sees the document tokens).
A separate injection detector is trained on the injection train documents (label: any
injection) and tested on held-out documents.
"""
import os

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, FEATS, RESULTS, load_jsonl, save_json, seed_all

seed_all(0)
BAND = [int(x) for x in os.environ.get("BAND", ",".join(map(str, range(8, 21)))).split(",")]
CS = [0.001, 0.01, 0.1, 1.0, 10.0]
FPR_TARGET = 0.05

F = {n: torch.load(f"{FEATS}/{n}.pt", weights_only=False) for n in ("main", "xstest", "jbb", "injection")}
R = {n: load_jsonl(f"{DATA}/{n}.jsonl") for n in F}
layers, code_layers = F["main"]["layers"], F["main"]["code_layers"]
words, lex_names, dict_ids = F["main"]["words"], F["main"]["lex_names"], F["main"]["dict_ids"]
band_idx = [layers.index(l) for l in BAND]
# codes store the model-layer index (position in `layers`), see jlens/features.py
code_band = [layers.index(l) for l in BAND if l in code_layers]
M = len(dict_ids)
from transformers import AutoTokenizer  # noqa: E402
from common import QWEN  # noqa: E402
tok = AutoTokenizer.from_pretrained(QWEN)


def labels(name, key="label"):
    return np.array([r[key] for r in R[name]])


def split_mask(name, s):
    return np.array([r.get("split") == s for r in R[name]])


# ---- feature builders ---------------------------------------------------------------------
def probe_X(name, lens):
    return F[name]["probe"][lens][:, band_idx].reshape(len(R[name]), -1)


def codes_X(name, lens, pools=(0, 1, 2)):
    rows, cols, vals = [], [], []
    for i, c in enumerate(F[name]["codes"][lens]):
        keep = np.isin(c[:, 0].astype(int), code_band) & np.isin(c[:, 1].astype(int), pools)
        c = c[keep]
        rows.extend([i] * len(c))
        cols.extend((c[:, 1].astype(int) * M + c[:, 2].astype(int)).tolist())
        vals.extend(c[:, 3].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(R[name]), 3 * M))


def raw_X(name, layer):
    return F[name]["raw"][:, layers.index(layer)][:, [0, 2]].reshape(len(R[name]), -1).astype(np.float32)


def zero_shot(name, lens, lex, positions="span+tail"):
    """max over span (+tail) tokens of the band-averaged lexicon score."""
    li = lex_names.index(lex)
    out = []
    for i, t in enumerate(F[name]["tokens"][lens]):
        pos = F[name]["span"][i]
        if positions == "span+tail":
            pos = np.concatenate([pos, F[name]["tail"][i]])
        s = t[pos][:, band_idx, li].astype(np.float32).mean(1)
        out.append(s.max())
    return np.array(out)


# ---- metrics --------------------------------------------------------------------------------
def threshold_at_fpr(scores_neg, fpr=FPR_TARGET):
    return float(np.quantile(scores_neg, 1 - fpr))


def metrics(y, s, thr):
    pred = s >= thr
    return {"auroc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
            "tpr": float(pred[y == 1].mean()) if (y == 1).any() else None,
            "fpr": float(pred[y == 0].mean()) if (y == 0).any() else None,
            "acc": float((pred == (y == 1)).mean()), "n": int(len(y))}


def eval_sets():
    ipi_y = labels("injection", "label_harm")
    return {"main_test": ("main", split_mask("main", "test"), labels("main")),
            "xstest": ("xstest", np.ones(len(R["xstest"]), bool), labels("xstest")),
            "jbb": ("jbb", np.ones(len(R["jbb"]), bool), labels("jbb")),
            "ipi_harm": ("injection", np.ones(len(R["injection"]), bool), ipi_y)}


def fit_lr(Xtr, ytr, Xva, yva, dense=True):
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=3000)
        model = make_pipeline(StandardScaler(), clf) if dense else clf
        model.fit(Xtr, ytr)
        auc = roc_auc_score(yva, model.decision_function(Xva))
        if best is None or auc > best[0] + 1e-6:
            best = (auc, C, model)
    return best


tr, va = split_mask("main", "train"), split_mask("main", "val")
y = labels("main")
results = {"band": BAND, "fpr_target": FPR_TARGET, "methods": {}}
SCORES = {}


def run(name, build, dense=True, trained=True, sign=1.0):
    """build(dataset_name) -> features or scores."""
    entry = {}
    if trained:
        Xm = build("main")
        val_auc, C, model = fit_lr(Xm[tr], y[tr], Xm[va], y[va], dense)
        score = lambda n: model.decision_function(build(n))  # noqa: E731
        entry.update(C=C, val_auroc=float(val_auc))
    else:
        score = lambda n: sign * build(n)  # noqa: E731
        model = None
        entry["val_auroc"] = float(roc_auc_score(y[va], score("main")[va]))
    s_main = score("main")
    thr = threshold_at_fpr(s_main[va & (y == 0)])
    entry["threshold"] = thr
    SCORES[name] = {"threshold": thr, **{ds: score(ds).tolist() for ds in R}}
    for set_name, (ds, mask, yy) in eval_sets().items():
        s = s_main if ds == "main" else score(ds)
        entry[set_name] = metrics(yy[mask], s[mask], thr)
    results["methods"][name] = entry
    print(f"{name:42s} " + "  ".join(
        f"{k}: auc={entry[k]['auroc']:.3f} tpr={entry[k]['tpr']:.2f} fpr={entry[k]['fpr']:.2f}"
        for k in eval_sets()), flush=True)
    return model


# J-space (ours) and its logit-lens ablation
m_codes = run("J-space codes + LR", lambda n: codes_X(n, "jlens"), dense=False)
run("Logit-space codes + LR (ablation)", lambda n: codes_X(n, "logit"), dense=False)
run("J-lens concept probe + LR", lambda n: probe_X(n, "jlens"))
run("Logit-lens concept probe + LR (ablation)", lambda n: probe_X(n, "logit"))
for lex in ("harm", "refusal", "injection", "random"):
    run(f"Zero-shot J-lens '{lex}' lexicon", lambda n, lex=lex: zero_shot(n, "jlens", lex), trained=False)
    run(f"Zero-shot logit-lens '{lex}' lexicon", lambda n, lex=lex: zero_shot(n, "logit", lex), trained=False)

# baselines: raw residual probe at every layer (best on val reported), TF-IDF on the text
raw_val = {}
for l in layers:
    Xm = raw_X("main", l)
    raw_val[l] = fit_lr(Xm[tr], y[tr], Xm[va], y[va])[0]
best_layer = max(raw_val, key=raw_val.get)
results["raw_probe_val_auroc_per_layer"] = {int(k): float(v) for k, v in raw_val.items()}
run(f"Raw residual probe + LR (layer {best_layer})", lambda n: raw_X(n, best_layer))
texts = {n: [r["text"] for r in R[n]] for n in R}
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit([texts["main"][i] for i in np.where(tr)[0]])
run("TF-IDF + LR (text only)", lambda n: tfidf.transform(texts[n]), dense=False)

# per-layer zero-shot curves (J-lens vs logit lens, harm lexicon), main val+test
curves = {}
for lens in ("jlens", "logit"):
    li = lex_names.index("harm")
    per_layer = []
    for j in range(len(layers)):
        s = np.array([t[np.concatenate([F["main"]["span"][i], F["main"]["tail"][i]])][:, j, li].astype(
            np.float32).max() for i, t in enumerate(F["main"]["tokens"][lens])])
        per_layer.append(float(roc_auc_score(y[~tr], s[~tr])))
    curves[lens] = per_layer
results["zero_shot_harm_auroc_per_layer"] = curves

# interpretation: J-space atoms with the largest weights
coef = m_codes.coef_[0]
pool_names = ["mean_span", "mean_tail", "last"]
top = {}
for sign, name in ((1, "malicious"), (-1, "benign")):
    order = np.argsort(-sign * coef)[:40]
    top[name] = [{"pool": pool_names[c // M], "token": tok.decode([dict_ids[c % M]]), "weight": float(coef[c])}
                 for c in order]
results["jspace_top_atoms"] = top
print("top malicious atoms:", [t["token"] for t in top["malicious"][:25]])
print("top benign atoms:", [t["token"] for t in top["benign"][:25]])

# most frequent J-space atoms (mean span pool, band layers) per class, for the paper table
def atom_freq(name, mask, lens="jlens", pool=0, k=15):
    cnt = np.zeros(M)
    for i in np.where(mask)[0]:
        c = F[name]["codes"][lens][i]
        c = c[np.isin(c[:, 0].astype(int), code_band) & (c[:, 1].astype(int) == pool)]
        np.add.at(cnt, c[:, 2].astype(int), c[:, 3])
    order = np.argsort(-cnt)[:k]
    return [(tok.decode([dict_ids[a]]), float(cnt[a] / mask.sum())) for a in order]


ipi_variant = np.array([r["variant"] for r in R["injection"]])
results["frequent_atoms"] = {
    "advbench (span)": atom_freq("main", y == 1),
    "alpaca (span)": atom_freq("main", y == 0),
    "advbench (tail)": atom_freq("main", y == 1, pool=1),
    "alpaca (tail)": atom_freq("main", y == 0, pool=1),
    "ipi inj_harm (doc)": atom_freq("injection", ipi_variant == "inj_harm"),
    "ipi inj_benign (doc)": atom_freq("injection", ipi_variant == "inj_benign"),
    "ipi clean (doc)": atom_freq("injection", ipi_variant == "clean"),
}
for k, v in results["frequent_atoms"].items():
    print(f"{k:22s}", ", ".join(w.strip() for w, _ in v))

# ---- injection detector (trained on injection train docs, any injection = positive) ----
yi = labels("injection", "label_inj")
itr, iva, ite = (split_mask("injection", s) for s in ("train", "val", "test"))
inj = {}
for name, build, dense in (("J-space codes + LR", lambda: codes_X("injection", "jlens", pools=(0,)), False),
                           ("Logit-space codes + LR", lambda: codes_X("injection", "logit", pools=(0,)), False),
                           ("J-lens concept probe + LR", lambda: probe_X("injection", "jlens"), True),
                           (f"Raw residual probe (layer {best_layer})", lambda: raw_X("injection", best_layer), True)):
    X = build()
    auc, C, model = fit_lr(X[itr], yi[itr], X[iva], yi[iva], dense)
    s = model.decision_function(X)
    inj[name] = {"val_auroc": float(auc), "test_auroc": float(roc_auc_score(yi[ite], s[ite])),
                 "test_auroc_benign_injections_vs_clean": float(roc_auc_score(
                     yi[ite & (ipi_variant != "inj_harm")], s[ite & (ipi_variant != "inj_harm")]))}
    if name.startswith("J-space"):
        coef = model.coef_[0][:M]
        inj["top_atoms"] = [tok.decode([dict_ids[a]]) for a in np.argsort(-coef)[:30]]
    print("injection detector", name, inj[name])
for lex in ("injection", "harm", "random"):
    s = zero_shot("injection", "jlens", lex, positions="span")
    inj[f"Zero-shot J-lens '{lex}' lexicon"] = {
        "test_auroc": float(roc_auc_score(yi[ite], s[ite])), "all_auroc": float(roc_auc_score(yi, s))}
results["injection_detector"] = inj
print("injection top atoms:", inj.get("top_atoms"))

# ---- localization: are injected tokens the ones that light up? ---------------------------------
def token_offsets(r):
    msgs = [{"role": "user", "content": r["text"]}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    base = rendered.index(r["text"])
    enc = tok(rendered, return_offsets_mapping=True)
    return base, enc["offset_mapping"]


loc = {}
for lex in ("injection", "harm", "random"):
    li = lex_names.index(lex)
    ys, ss = [], []
    for i, r in enumerate(R["injection"]):
        if "inj_span" not in r:
            continue
        base, offs = token_offsets(r)
        a, b = base + r["inj_span"][0], base + r["inj_span"][1]
        t = F["injection"]["tokens"]["jlens"][i][:, band_idx, li].astype(np.float32).mean(1)
        for p in F["injection"]["span"][i]:
            s0, e0 = offs[p]
            ys.append(int(s0 < b and e0 > a))
            ss.append(t[p])
    loc[lex] = float(roc_auc_score(ys, ss))
results["token_localization_auroc"] = loc
print("token-level localization AUROC (injected tokens vs other document tokens):", loc)

save_json(results, f"{RESULTS}/04_classification.json")
torch.save({"scores": SCORES, "ids": {n: [r["id"] for r in R[n]] for n in R}}, f"{FEATS}/04_scores.pt")

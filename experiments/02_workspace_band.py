"""Where is the workspace in Qwen3.5-0.8B? J-lens vs logit lens, layer by layer.

1. Lens quality on Anthropic's released lens-eval sets (multihop: the bridge entity, e.g.
   'Brazil' for "the ocean on the coast of the country where Carnival is celebrated";
   association: an unnamed concept, e.g. 'grief'). For each item we take the rank of the
   intermediate in the lens readout at the evaluation position, per layer, and report
   pass@k of the min-over-layers rank (the paper's metric) plus per-layer pass@10.
2. Layer signatures on held-out WikiText (paper Fig. 28):
   (a) agreement of the lens top-1 token with the model's actual next-token prediction,
   (b) excess kurtosis of the lens logit distribution (how peaked / non-random it is),
   (c) persistence: P(top-1 at t == top-1 at t+d) minus a position-shuffled null.
3. Mean residual per layer on WikiText (used later to center activations).
"""
import json
import os

import numpy as np
import torch

from common import MODELS, QWEN, QWEN_LENS, RESULTS, save_json, wikitext
from jlens import JacobianLens, ModelAdapter

torch.set_num_threads(4)
EVAL_DIR = os.environ.get("LENS_EVAL_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "lens_eval"))

adapter = ModelAdapter(QWEN)
lens = JacobianLens.load(adapter, QWEN_LENS)
layers = lens.layers
logit = JacobianLens.identity(adapter, layers)
means_path = f"{MODELS}/lenses/qwen3.5-0.8b_means.pt"
if not os.path.exists(means_path):
    lens.fit_means(wikitext(200, offset=1000))
    torch.save({l: m.cpu() for l, m in lens.means.items()}, means_path)
print("layers", layers, flush=True)


def first_ids(word):
    ids = set()
    for w in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        t = adapter.tokenizer.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


@torch.no_grad()
def ranks_at(prompt, words, pos=-1):
    """rank (0 = top) of each word, per layer, for both lenses -> dict lens -> (L, n_words)."""
    ids = adapter.tokenize(prompt)
    acts = adapter.activations(ids, layers)
    out = {}
    for name, ln in (("jlens", lens), ("logit", logit)):
        R = np.zeros((len(layers), len(words)))
        for i, l in enumerate(layers):
            lg = ln.logits(acts[l][0, pos], l).float()
            for j, w in enumerate(words):
                best = lg[first_ids(w)].max()
                R[i, j] = int((lg > best).sum())
        out[name] = R
    return out


res = {"layers": layers, "evals": {}}
KS = [1, 5, 10, 50, 100]
for name in ("multihop", "association"):
    items = json.load(open(os.path.join(EVAL_DIR, f"lens-eval-{name}.json")))["items"]
    per = {"jlens": [], "logit": []}
    for it in items:
        r = ranks_at(it["prompt"], it["intermediates"])
        for k in per:
            per[k].append(r[k])
    summary = {}
    for k, Rs in per.items():
        mins = np.concatenate([R.min(0) for R in Rs])                   # min over layers
        per_layer = np.stack([(R < 10).mean(1) for R in Rs]).mean(0)    # pass@10 per layer
        summary[k] = {"pass@k_min_over_layers": {str(kk): float((mins < kk).mean()) for kk in KS},
                      "pass@10_per_layer": per_layer.tolist()}
    res["evals"][name] = summary
    print(name, {k: v["pass@k_min_over_layers"] for k, v in summary.items()}, flush=True)

save_json(res, f"{RESULTS}/02_workspace_band_qwen.json")

# ---- layer signatures on WikiText ---------------------------------------------------------
texts = wikitext(int(os.environ.get("N_SIG", 8)), offset=1300)
sig = {k: {"agree_top1": [], "kurtosis": [], "persistence": []} for k in ("jlens", "logit")}
acc = {k: np.zeros((3, len(layers))) for k in sig}
counts = np.zeros(len(layers))
rng = np.random.default_rng(0)
with torch.no_grad():
    for text in texts:
        ids = adapter.tokenize(text, 128)
        if ids.shape[1] < 64:
            continue
        logits, acts = adapter.forward_with_activations(ids, layers)
        pos = torch.arange(16, ids.shape[1] - 1)
        model_top = logits[pos].argmax(-1)
        for name, ln in (("jlens", lens), ("logit", logit)):
            for i, l in enumerate(layers):
                lg = ln.logits(acts[l][pos], l).float()
                top = lg.argmax(-1)
                z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
                kurt = float((z ** 4).mean(-1).mean() - 3)
                agree = float((top == model_top).float().mean())
                tt = top.cpu().numpy()
                same = np.mean([np.mean(tt[d:] == tt[:-d]) for d in range(1, 9)])
                shuf = rng.permutation(tt)
                null = np.mean([np.mean(shuf[d:] == shuf[:-d]) for d in range(1, 9)])
                acc[name][:, i] += [agree, kurt, same - null]
        counts += 1
for name in sig:
    a = acc[name] / counts
    sig[name] = {"agree_top1": a[0].tolist(), "kurtosis": a[1].tolist(), "persistence": a[2].tolist()}
res["signatures"] = sig
save_json(res, f"{RESULTS}/02_workspace_band_qwen.json")
for name in sig:
    print(name)
    for key, vals in sig[name].items():
        print(f"  {key:12s}", " ".join(f"{v:6.2f}" for v in vals))

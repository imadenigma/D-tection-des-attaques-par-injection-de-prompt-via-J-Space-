"""Detector-gated J-space steering as a defense.

1. Strength sweep on the *val* split (always-on steering, 8 harmful + 8 benign prompts):
   the smallest alpha that makes the model refuse >= 90% of harmful val prompts is kept.
2. Evaluation (greedy decoding, 32 new tokens) under three conditions:
     none    the undefended model,
     gated   steering applied only when the J-space detector flags the prompt
             (threshold fixed on val for 5% FPR in 04_classify.py),
     always  steering applied to every prompt (shows what gating buys on benign prompts).
   Outcome: refusal rate (prefix/keyword judge, see jlens/defense.py).
Texts generated for harmful prompts are not saved; only refusal flags are.
"""
import os
import random
import time

import numpy as np
import torch

from common import DATA, FEATS, QWEN, QWEN_LENS, RESULTS, load_jsonl, save_json
from jlens import JacobianLens, ModelAdapter
from jlens.defense import SteeringDefense, is_refusal

torch.set_num_threads(4)
DETECTOR = os.environ.get("DETECTOR", "J-space codes + LR")
STEER_LAYERS = [int(x) for x in os.environ.get("STEER_LAYERS", "8,10,12,14,16,18,20").split(",")]
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.1,0.2,0.4").split(",")]
MAX_NEW = 32
rng = random.Random(0)

adapter = ModelAdapter(QWEN)
lens = JacobianLens.load(adapter, QWEN_LENS)
sc = torch.load(f"{FEATS}/04_scores.pt", weights_only=False)
scores, thr = sc["scores"][DETECTOR], sc["scores"][DETECTOR]["threshold"]
score_of = {i: s for n in sc["ids"] for i, s in zip(sc["ids"][n], scores[n])}

main = load_jsonl(f"{DATA}/main.jsonl")
inj = load_jsonl(f"{DATA}/injection.jsonl")
xs = load_jsonl(f"{DATA}/xstest.jsonl")
pick = lambda rows, n: rng.sample(rows, min(n, len(rows)))  # noqa: E731

defense = SteeringDefense(adapter, lens, STEER_LAYERS)
defense.calibrate_norms([adapter.chat(r["text"]) for r in main if r["split"] == "train"][:20])

# ---- 1. strength sweep on val -------------------------------------------------------------------
val_h = pick([r for r in main if r["split"] == "val" and r["label"] == 1], 8)
val_b = pick([r for r in main if r["split"] == "val" and r["label"] == 0], 8)
sweep = {}
for a in [0.0] + ALPHAS:
    rh = np.mean([is_refusal(defense.generate(adapter.chat(r["text"]), steer=a > 0, alpha=a, max_new_tokens=MAX_NEW))
                  for r in val_h])
    outs_b = [defense.generate(adapter.chat(r["text"]), steer=a > 0, alpha=a, max_new_tokens=MAX_NEW) for r in val_b]
    sweep[a] = {"refusal_harmful": float(rh), "refusal_benign": float(np.mean([is_refusal(o) for o in outs_b])),
                "benign_example": outs_b[0][:200]}
    print("alpha", a, sweep[a], flush=True)
ok = [a for a in ALPHAS if sweep[a]["refusal_harmful"] >= 0.9]
alpha = min(ok) if ok else max(ALPHAS, key=lambda a: sweep[a]["refusal_harmful"])
print("chosen alpha", alpha, flush=True)

# ---- 2. evaluation ------------------------------------------------------------------------------
test_docs = [r for r in inj if r["split"] == "test"]
sets = {
    "advbench_test": (pick([r for r in main if r["split"] == "test" and r["label"] == 1], 50), True),
    "alpaca_test": (pick([r for r in main if r["split"] == "test" and r["label"] == 0], 50), False),
    "ipi_inj_harm": ([r for r in test_docs if r["variant"] == "inj_harm"], True),
    "ipi_clean": ([r for r in test_docs if r["variant"] == "clean"], False),
    "xstest_safe": (pick([r for r in xs if r["label"] == 0], 40), False),
    "xstest_unsafe": (pick([r for r in xs if r["label"] == 1], 30), True),
}
ALWAYS_N = 15
res = {"detector": DETECTOR, "threshold": thr, "steer_layers": STEER_LAYERS, "alpha": alpha, "sweep": sweep,
       "layer_norms": defense.layer_norms, "max_new_tokens": MAX_NEW, "sets": {}}
t0 = time.time()
for name, (rows, harmful) in sets.items():
    items = []
    for j, r in enumerate(rows):
        prompt = adapter.chat(r["text"])
        flagged = score_of[r["id"]] >= thr
        base = defense.generate(prompt, steer=False, max_new_tokens=MAX_NEW)
        gated = defense.generate(prompt, steer=True, alpha=alpha, max_new_tokens=MAX_NEW) if flagged else base
        it = {"id": r["id"], "flagged": bool(flagged), "refusal_none": is_refusal(base),
              "refusal_gated": is_refusal(gated)}
        if j < ALWAYS_N:
            always = gated if flagged else defense.generate(prompt, steer=True, alpha=alpha, max_new_tokens=MAX_NEW)
            it["refusal_always"] = is_refusal(always)
            if not harmful:
                it["text_always"] = always[:200]
        if not harmful:
            it["text_none"] = base[:200]
        items.append(it)
    summ = {k: float(np.mean([it[k] for it in items if k in it]))
            for k in ("flagged", "refusal_none", "refusal_gated", "refusal_always")}
    summ["n"] = len(items)
    res["sets"][name] = {"harmful": harmful, "summary": summ, "items": items}
    print(name, summ, f"{time.time() - t0:.0f}s", flush=True)
    save_json(res, f"{RESULTS}/05_defense.json")

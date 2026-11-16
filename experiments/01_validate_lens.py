"""Check our Jacobian-lens estimator against a published lens.

Fits J_l on Pythia-70m (CPU-friendly) with `jlens.jacobian` on WikiText and compares it
with the lens Neuronpedia fitted with Anthropic's reference code (1000 WikiText prompts):
  * cosine similarity between the flattened J_l matrices,
  * agreement of the top-10 lens readouts on held-out text.
Also reports, as a sanity check, how far each J_l is from the identity (logit lens).
"""
import time

import torch

from common import MODELS, RESULTS, save_json, seed_all, wikitext
from jlens import JacobianLens, ModelAdapter
from jlens.jacobian import jacobians_for_prompt

N_FIT = int(__import__("os").environ.get("N_FIT", 60))

seed_all(0)
torch.set_num_threads(4)
adapter = ModelAdapter(f"{MODELS}/pythia-70m-deduped")
published = JacobianLens.load(adapter, f"{MODELS}/lenses/pythia-70m-deduped/jlens/Salesforce-wikitext/"
                                       "pythia-70m-deduped_jacobian_lens.pt")
layers = published.layers
CHECKPOINTS = [1, 2, 5, 10, 20, 40, 80]
t0 = time.time()
total, n, convergence = None, 0, []
for text in wikitext(N_FIT):
    J, _ = jacobians_for_prompt(adapter, text, layers, row_batch=64, max_len=128)
    total = J if total is None else {l: total[l] + J[l] for l in layers}
    n += 1
    if n in CHECKPOINTS or n == N_FIT:
        cos = {l: float(torch.nn.functional.cosine_similarity(
            (total[l] / n).flatten(), published.J[l].flatten(), dim=0)) for l in layers}
        convergence.append({"n_prompts": n, "cos_per_layer": cos})
        print("n", n, {l: round(c, 3) for l, c in cos.items()}, flush=True)
fit_time = time.time() - t0
ours = JacobianLens(adapter, {l: total[l] / n for l in layers}, n_prompts=n)
ours.save(f"{MODELS}/lenses/pythia70m_ours_n{N_FIT}.pt")

held_out = wikitext(30, offset=1000)
res = {"model": "EleutherAI/pythia-70m-deduped", "n_fit_prompts": ours.n_prompts,
       "fit_seconds": fit_time, "published_n_prompts": published.n_prompts, "convergence": convergence, "layers": {}}
eye = torch.eye(adapter.d_model)
for l in layers:
    a, b = ours.J[l].flatten(), published.J[l].flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    cos_eye_pub = float(torch.nn.functional.cosine_similarity(eye.flatten(), b, dim=0))
    overlaps, top1 = [], []
    for text in held_out:
        ids = adapter.tokenize(text, 128)
        h = adapter.activations(ids, [l])[l][0, 16:-1]
        ta = ours.logits(h, l).topk(10).indices
        tb = published.logits(h, l).topk(10).indices
        overlaps.append(float((ta[:, :, None] == tb[:, None, :]).any(-1).float().mean()))
        top1.append(float((ta[:, 0] == tb[:, 0]).float().mean()))
    res["layers"][l] = {"cos_J_ours_vs_published": cos, "cos_I_vs_published": cos_eye_pub,
                        "top10_overlap": sum(overlaps) / len(overlaps), "top1_agreement": sum(top1) / len(top1)}
    print(l, res["layers"][l], flush=True)
save_json(res, f"{RESULTS}/01_validate_lens_pythia70m.json")
print("fit seconds", fit_time)

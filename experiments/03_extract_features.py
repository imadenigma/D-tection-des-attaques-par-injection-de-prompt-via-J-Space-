"""Extract J-space / logit-lens / raw-activation features for every prompt of every dataset.

Outputs one file per dataset in $JSPACE_WORK/features/<name>.pt (not versioned: large).
Re-running skips datasets that are already done.
"""
import os
import sys
import time

import numpy as np
import torch

from common import DATA, FEATS, MODELS, QWEN, QWEN_LENS, load_jsonl
from jlens import JacobianLens, ModelAdapter
from jlens.features import Featurizer

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
CODE_LAYERS = [int(x) for x in os.environ.get("CODE_LAYERS", "8,11,14,17,20").split(",")]
BATCH = 32
DATASETS = sys.argv[1:] or ["main", "injection", "xstest", "jbb"]

adapter = ModelAdapter(QWEN)
lens = JacobianLens.load(adapter, QWEN_LENS)
means = torch.load(f"{MODELS}/lenses/qwen3.5-0.8b_means.pt", weights_only=True)
lens.means = {l: m.to(adapter.device) for l, m in means.items()}
logit = JacobianLens.identity(adapter, lens.layers, means)
fz = Featurizer(adapter, lens, logit, CODE_LAYERS, k=int(os.environ.get("K", 16)), inner_steps=2)
print(f"concept words: {len(fz.words)}  dictionary atoms: {len(fz.dict_ids)}  code layers: {CODE_LAYERS}",
      flush=True)

for name in DATASETS:
    out_path = f"{FEATS}/{name}.pt"
    if os.path.exists(out_path):
        print("skip", name)
        continue
    rows = load_jsonl(f"{DATA}/{name}.jsonl")
    feats, t0 = [], time.time()
    for b in range(0, len(rows), BATCH):
        batch = [fz(r["text"], span=r.get("doc_span")) for r in rows[b:b + BATCH]]
        feats.extend(fz.add_codes(batch))
        el = time.time() - t0
        print(f"[{name}] {len(feats)}/{len(rows)}  {el / len(feats):.2f}s/prompt", flush=True)
    torch.save({
        "ids": [r["id"] for r in rows],
        "words": fz.words, "lex_names": fz.lex_names, "dict_ids": fz.dict_ids,
        "layers": lens.layers, "code_layers": CODE_LAYERS,
        "n_tokens": np.array([f.n_tokens for f in feats]),
        "span": [f.span for f in feats], "tail": [f.tail for f in feats],
        "raw": np.stack([f.raw for f in feats]),
        "probe": {k: np.stack([f.probe[k] for f in feats]) for k in ("jlens", "logit")},
        "tokens": {k: [f.tokens[k] for f in feats] for k in ("jlens", "logit")},
        "codes": {k: [f.codes[k] for f in feats] for k in ("jlens", "logit")},
        "next_token_top": np.stack([f.next_token_top for f in feats]),
    }, out_path)
    print("saved", out_path, flush=True)

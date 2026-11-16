"""Shared paths and helpers for the experiment scripts."""
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WORK = os.environ.get("JSPACE_WORK", "/home/user/work")          # large files (models, features)
MODELS = os.path.join(WORK, "models")
FEATS = os.path.join(WORK, "features")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figures")

QWEN = os.environ.get("JSPACE_MODEL", os.path.join(MODELS, "Qwen3.5-0.8B"))
QWEN_LENS = os.environ.get("JSPACE_LENS", os.path.join(
    MODELS, "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"))

for p in (FEATS, RESULTS, FIGS):
    os.makedirs(p, exist_ok=True)


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def wikitext(n=None, offset=0):
    rows = json.load(open(os.path.join(WORK, "data_raw", "wikitext.json")))
    rows = [r["text"] for r in rows][offset:]
    return rows[:n] if n else rows

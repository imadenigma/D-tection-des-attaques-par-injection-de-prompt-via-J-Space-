"""J-space detector of malicious prompts.

A prompt is represented by its J-space coordinates: the nonnegative sparse decomposition
of its (centered, pooled) residual activations over the dictionary of J-lens vectors of
word tokens, at the workspace layers (see `features.py`). Coordinates are summed over
layers, giving one sparse, human-readable vector per prompt ("which words are in the
workspace, and how strongly"). A logistic regression on that vector is the detector; its
weights are themselves words, so each decision can be explained by the atoms that drove it.
"""
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .features import CODE_POOLS


class JSpaceDetector:
    def __init__(self, featurizer, band=None, pools=(0, 1, 2), C=0.1, threshold=0.0):
        self.fz = featurizer
        self.band = list(band or featurizer.code_layers)
        self.pools = tuple(pools)
        self.C = C
        self.threshold = threshold
        self.clf = None

    # ---- features ------------------------------------------------------------------------
    def featurize(self, texts, spans=None, batch=32):
        spans = spans or [None] * len(texts)
        feats = []
        for b in range(0, len(texts), batch):
            chunk = [self.fz(t, span=s) for t, s in zip(texts[b:b + batch], spans[b:b + batch])]
            feats.extend(self.fz.add_codes(chunk))
        return feats

    def matrix(self, feats):
        M = len(self.fz.dict_ids)
        band = [self.fz.layers.index(l) for l in self.band]  # codes store model-layer indices
        rows, cols, vals = [], [], []
        for i, f in enumerate(feats):
            c = f.codes["jlens"]
            c = c[np.isin(c[:, 0].astype(int), band) & np.isin(c[:, 1].astype(int), self.pools)]
            rows.extend([i] * len(c))
            cols.extend((c[:, 1].astype(int) * M + c[:, 2].astype(int)).tolist())
            vals.extend(c[:, 3].tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(feats), len(CODE_POOLS) * M))

    # ---- training / inference --------------------------------------------------------------
    def fit(self, texts, labels, spans=None):
        X = self.matrix(self.featurize(texts, spans))
        self.clf = LogisticRegression(C=self.C, max_iter=3000).fit(X, np.asarray(labels))
        return self

    def score(self, texts, spans=None):
        return self.clf.decision_function(self.matrix(self.featurize(texts, spans)))

    def explain(self, text, span=None, n=10):
        """Atoms (pool, word, contribution) that pushed this prompt's score the most."""
        X = self.matrix(self.featurize([text], [span])).tocoo()
        M = len(self.fz.dict_ids)
        contrib = [(CODE_POOLS[c // M], self.fz.adapter.decode(self.fz.dict_ids[c % M]),
                    float(v * self.clf.coef_[0][c])) for c, v in zip(X.col, X.data)]
        agg = {}
        for pool, w, v in contrib:
            agg[(pool, w)] = agg.get((pool, w), 0.0) + v
        return sorted(((p, w, v) for (p, w), v in agg.items()), key=lambda x: -abs(x[2]))[:n]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"band": self.band, "pools": self.pools, "C": self.C, "threshold": self.threshold,
                         "clf": self.clf, "dict_ids": self.fz.dict_ids}, f)

    @classmethod
    def load(cls, featurizer, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["dict_ids"] != featurizer.dict_ids:
            raise ValueError("detector was trained with a different dictionary")
        det = cls(featurizer, state["band"], state["pools"], state["C"], state["threshold"])
        det.clf = state["clf"]
        return det

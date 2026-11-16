"""J-space: the set of activations writable as a sparse nonnegative combination
of at most k J-lens vectors,

    J-space_k = { sum_{i in S} c_i v_i  :  |S| <= k,  c_i >= 0 }.

It is a union of low-dimensional cones, not a vector space. For an activation
h, the J-space component is the closest point of this set and the non-J-space
component is the remainder:

    h = h_J + h_perp,   h_J = argmin_{x in J-space_k} ||h - x||.

The coefficients c are the local J-space coordinates ("what is in the
workspace right now"). The projection is approximated with nonnegative
gradient pursuit (Blumensath & Davies, 2008) over unit-normalized atoms.

Choosing k: atoms are added while each one explains more residual energy than
a random direction would. A uniformly random unit direction captures
||r||^2 / d of the residual in expectation, which is the control used here.
"""
import re
from dataclasses import dataclass

import torch


def word_token_ids(tokenizer, min_len=3, leading_space=True):
    """Ids of tokens that decode to a single alphabetic word with a leading space
    (word-initial tokens such as ' bomb', ' Hack'; `leading_space=False` also keeps the
    no-space variants). Used as the J-space dictionary: it keeps the readout
    human-readable and makes batched decomposition ~4x cheaper on large vocabularies
    (most of Qwen's 248k tokens are sub-words, punctuation or CJK)."""
    pat = re.compile(rf"^{' ' if leading_space else ' ?'}[A-Za-z]{{{min_len},}}$")
    strings = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    return [i for i, s in enumerate(strings) if pat.match(s)]


@torch.no_grad()
def batched_pursuit(H, D, k=25, inner_steps=3, adaptive=True, control_margin=1.0):
    """Nonnegative gradient pursuit for many activations at once.

    H: (N, d) activations; D: (M, d) dictionary of unit-norm atoms (J-lens vectors).
    Greedily adds the atom most positively correlated with the residual, then runs a
    few projected gradient steps (exact line search, coefficients kept >= 0) on the
    selected support. With `adaptive`, a sample stops as soon as the new atom removes
    less residual energy than a random unit direction would in expectation (||r||^2/d).

    Returns (S, C, R): S (N, k) selected dictionary indices (-1 = unused), C (N, k)
    nonnegative coefficients, R (N, d) residuals h - h_J.
    """
    H = H.float()
    N, d = H.shape
    S = torch.full((N, k), -1, dtype=torch.long, device=H.device)
    C = torch.zeros(N, k, device=H.device)
    R = H.clone()
    active = torch.ones(N, dtype=torch.bool, device=H.device)
    for it in range(k):
        corr = R @ D.T
        if it:
            prev = S[:, :it]
            corr.scatter_(1, prev.clamp_min(0), float("-inf"))
        best, idx = corr.max(1)
        newly = active & (best > 0)
        S[:, it] = torch.where(newly, idx, torch.full_like(idx, -1))
        sel = S[:, :it + 1]
        A = D[sel.clamp_min(0)] * (sel >= 0).unsqueeze(-1).float()     # (N, s, d)
        c = C[:, :it + 1].clone()
        r = R
        energy = (R * R).sum(1)
        for _ in range(inner_steps):
            g = torch.bmm(A, r.unsqueeze(-1)).squeeze(-1)               # -grad of 0.5||h - A^T c||^2
            g = torch.where((c <= 0) & (g < 0), torch.zeros_like(g), g)
            Ag = torch.bmm(A.transpose(1, 2), g.unsqueeze(-1)).squeeze(-1)
            denom = (Ag * Ag).sum(1)
            step = (g * g).sum(1) / denom.clamp_min(1e-12)
            cap = torch.where(g < 0, c / (-g).clamp_min(1e-12), torch.full_like(c, float("inf"))).min(1).values
            step = torch.where(denom > 0, torch.minimum(step, cap), torch.zeros_like(step))
            c = (c + step.unsqueeze(-1) * g).clamp_min(0)
            r = H - torch.bmm(A.transpose(1, 2), c.unsqueeze(-1)).squeeze(-1)
        gain = energy - (r * r).sum(1)
        accept = newly & ((gain >= control_margin * energy / d) if adaptive else torch.ones_like(newly))
        C[:, :it + 1] = torch.where(accept.unsqueeze(-1), c, C[:, :it + 1])
        R = torch.where(accept.unsqueeze(-1), r, R)
        S[~accept, it] = -1
        active = accept
        if not active.any():
            break
    return S, C, R


@dataclass
class Decomposition:
    layer: int
    token_ids: list          # selected J-lens atoms, in selection order
    tokens: list             # decoded strings
    coeffs: torch.Tensor     # nonnegative coordinates on the *unit-normalized* atoms
    j_component: torch.Tensor
    residual: torch.Tensor
    explained: float         # 1 - ||h_perp||^2 / ||h||^2

    def items(self):
        order = self.coeffs.argsort(descending=True)
        return [(self.tokens[i], float(self.coeffs[i])) for i in order]

    def mass(self, token_ids):
        """Total coefficient on the given token ids."""
        wanted = set(int(t) for t in token_ids)
        return float(sum(c for t, c in zip(self.token_ids, self.coeffs.tolist()) if t in wanted))


class JSpace:
    def __init__(self, lens):
        self.lens = lens

    def _unit_atoms(self, layer, ids):
        return self.lens.vectors(layer, ids) / self.lens.atom_norms(layer)[ids][:, None]

    @torch.no_grad()
    def decompose(self, h, layer, k=25, inner_steps=3, adaptive=True, control_margin=1.0,
                  center=False, exclude=()):
        """Sparse nonnegative decomposition of one activation h: (d_model,)."""
        lens = self.lens
        h = h.float()
        if center and layer in lens.means:
            h = h - lens.means[layer]
        d = h.shape[0]
        norms = lens.atom_norms(layer)
        banned = torch.zeros_like(norms, dtype=torch.bool)
        if exclude:
            banned[torch.as_tensor(list(exclude), device=banned.device)] = True

        S, A = [], h.new_zeros(0, d)
        c = h.new_zeros(0)
        r = h.clone()

        for _ in range(k):
            corr = lens.scores(r, layer) / norms
            corr[banned] = -float("inf")
            tok = int(corr.argmax())
            if corr[tok] <= 0:
                break
            r_energy = float(r @ r)
            S_new = S + [tok]
            A_new = torch.cat([A, self._unit_atoms(layer, [tok])])
            c_new = torch.cat([c, c.new_zeros(1)])
            r_new = r

            for _ in range(inner_steps):
                g = A_new @ r_new                                 # -grad of 0.5||h - A^T c||^2
                g = torch.where((c_new <= 0) & (g < 0), torch.zeros_like(g), g)
                Ag = A_new.T @ g
                denom = float(Ag @ Ag)
                if denom <= 0:
                    break
                step = float(g @ g) / denom
                neg = g < 0
                if neg.any():                                     # stay in the nonnegative orthant
                    step = min(step, float((c_new[neg] / -g[neg]).min()))
                c_new = (c_new + step * g).clamp_min(0)
                r_new = h - A_new.T @ c_new

            gain = r_energy - float(r_new @ r_new)
            if adaptive and gain < control_margin * r_energy / d:
                break
            S, A, c, r = S_new, A_new, c_new, r_new
            banned[tok] = True

        h_energy = float(h @ h) or 1.0
        return Decomposition(
            layer=layer,
            token_ids=S,
            tokens=[lens.adapter.decode(t) for t in S],
            coeffs=c,
            j_component=h - r,
            residual=r,
            explained=1.0 - float(r @ r) / h_energy,
        )

    def decompose_sequence(self, hs, layer, **kw):
        """Decompose every position of hs: (seq, d_model)."""
        return [self.decompose(h, layer, **kw) for h in hs]

    @torch.no_grad()
    def occupancy(self, h, layer, k_max=50, **kw):
        """Number of J-lens atoms that beat the random-direction control (workspace occupancy)."""
        return len(self.decompose(h, layer, k=k_max, adaptive=True, **kw).token_ids)

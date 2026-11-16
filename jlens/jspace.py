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
from dataclasses import dataclass

import torch


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

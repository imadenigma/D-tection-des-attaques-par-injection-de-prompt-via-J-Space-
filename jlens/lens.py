"""The Jacobian lens (J-lens).

    LogitLens(h_l) = W_U norm(h_l)
    JLens(h_l)     = W_U norm(J_l h_l)

J_l corrects for the representational drift between layer l and the output,
so the J-lens gives meaningful readouts in middle layers where the logit lens
does not.

J-lens vectors: the row of W_U J_l for token v,

    v_tok = J_l^T W_U[tok]          (a direction in layer-l residual space)

encodes the potential for the model to verbalize `tok`, now or later. Pre-norm
lens logits are inner products <v_tok, h_l>. There are n_vocab such vectors in
d_model dimensions, so the set is overcomplete and non-orthogonal.
"""
import torch

from .jacobian import estimate_jacobians


class JacobianLens:
    def __init__(self, adapter, jacobians, means=None, n_prompts=None):
        self.adapter = adapter
        self.J = {l: J.to(adapter.device) for l, J in jacobians.items()}
        self.means = {l: m.to(adapter.device) for l, m in (means or {}).items()}
        self.n_prompts = n_prompts
        self._atom_norms = {}

    @property
    def layers(self):
        return sorted(self.J)

    # ---- construction / persistence -------------------------------------------------

    @classmethod
    def fit(cls, adapter, prompts, layers=None, **kw):
        layers = layers if layers is not None else adapter.workspace_layers()
        J, mu = estimate_jacobians(adapter, prompts, layers, **kw)
        return cls(adapter, J, mu, n_prompts=len(prompts))

    def save(self, path):
        torch.save({
            "model": self.adapter.name,
            "J": {l: J.cpu() for l, J in self.J.items()},
            "means": {l: m.cpu() for l, m in self.means.items()},
            "n_prompts": self.n_prompts,
        }, path)

    @classmethod
    def load(cls, adapter, path):
        state = torch.load(path, map_location="cpu")
        if state["model"] != adapter.name:
            raise ValueError(f"Lens fitted on {state['model']}, adapter is {adapter.name}")
        return cls(adapter, state["J"], state["means"], state["n_prompts"])

    # ---- readouts ------------------------------------------------------------------

    def _to_logits(self, x):
        W_U = self.adapter.W_U
        return self.adapter.final_norm(x.to(W_U.dtype)) @ W_U.T

    @torch.no_grad()
    def transport(self, h, layer):
        """J_l h: the activation carried to final-layer coordinates."""
        return h.float() @ self.J[layer].T

    @torch.no_grad()
    def logits(self, h, layer):
        """JLens(h_l) = W_U norm(J_l h_l). h: (..., d_model) -> (..., n_vocab)."""
        return self._to_logits(self.transport(h, layer))

    @torch.no_grad()
    def logit_lens(self, h):
        """Baseline LogitLens(h_l) = W_U norm(h_l)."""
        return self._to_logits(h)

    def top_tokens(self, h, layer, k=10, use_jacobian=True):
        """Top-k (token, prob) of the lens readout for a single activation h: (d_model,)."""
        logits = self.logits(h, layer) if use_jacobian else self.logit_lens(h)
        probs = logits.float().softmax(-1)
        p, idx = probs.topk(k)
        return [(self.adapter.decode(i), float(v)) for i, v in zip(idx, p)]

    # ---- J-lens vectors ---------------------------------------------------------------

    @torch.no_grad()
    def vectors(self, layer, token_ids):
        """J-lens vectors for `token_ids`: rows of W_U J_l, shape (len(ids), d_model)."""
        ids = torch.as_tensor(token_ids, device=self.adapter.device)
        return self.adapter.W_U[ids].float() @ self.J[layer]

    def vector(self, layer, word):
        return self.vectors(layer, [self.adapter.token_id(word)])[0]

    @torch.no_grad()
    def scores(self, h, layer):
        """<v_tok, h> for every token at once: W_U (J_l h). h: (..., d) -> (..., n_vocab)."""
        return self.transport(h, layer) @ self.adapter.W_U.float().T

    @torch.no_grad()
    def atom_norms(self, layer, chunk=8192):
        """||v_tok|| for all tokens (computed in chunks, cached)."""
        if layer not in self._atom_norms:
            W_U, J = self.adapter.W_U, self.J[layer]
            norms = torch.cat([
                (W_U[i:i + chunk].float() @ J).norm(dim=-1) for i in range(0, W_U.shape[0], chunk)
            ])
            self._atom_norms[layer] = norms.clamp_min(1e-8)
        return self._atom_norms[layer]

    def cosine(self, h, layer, word):
        """Per-concept probe: cos(v_word, h), comparable across layers and tokens."""
        v = self.vector(layer, word)
        return torch.nn.functional.cosine_similarity(h.float(), v.expand_as(h.float()), dim=-1)

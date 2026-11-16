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

Setting J_l = I everywhere turns every method of this class into its logit-lens
counterpart (`JacobianLens.identity`), which we use as an ablation.
"""
import torch

from .jacobian import estimate_jacobians, estimate_means


class JacobianLens:
    def __init__(self, adapter, jacobians, means=None, n_prompts=None, identity=False):
        self.adapter = adapter
        self.J = {int(l): J.float().to(adapter.device) for l, J in jacobians.items()}
        self.means = {int(l): m.float().to(adapter.device) for l, m in (means or {}).items()}
        self.n_prompts = n_prompts
        self.is_identity = identity
        self._atom_norms = {}

    @property
    def layers(self):
        return sorted(self.J)

    # ---- construction / persistence -------------------------------------------------

    @classmethod
    def fit(cls, adapter, prompts, layers=None, **kw):
        layers = layers if layers is not None else list(range(adapter.final_layer))
        J, n = estimate_jacobians(adapter, prompts, layers, **kw)
        return cls(adapter, J, n_prompts=n)

    @classmethod
    def identity(cls, adapter, layers, means=None):
        """The logit lens written as a J-lens with J_l = I (ablation baseline)."""
        eye = torch.eye(adapter.d_model)
        return cls(adapter, {l: eye for l in layers}, means, identity=True)

    def fit_means(self, prompts, **kw):
        self.means = {l: m.to(self.adapter.device) for l, m in
                      estimate_means(self.adapter, prompts, self.layers, **kw).items()}
        return self

    def save(self, path):
        torch.save({
            "model": self.adapter.name,
            "J": {l: J.cpu().half() for l, J in self.J.items()},
            "means": {l: m.cpu() for l, m in self.means.items()},
            "n_prompts": self.n_prompts,
            "source_layers": self.layers,
            "d_model": self.adapter.d_model,
        }, path)

    @classmethod
    def load(cls, adapter, path):
        """Load a lens saved by `save`, or a lens in the format of Anthropic's reference
        implementation / the Neuronpedia hub (keys J, n_prompts, source_layers, d_model)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in state:
            raise ValueError(f"{path} is not a lens file (keys {sorted(state)})")
        if state.get("d_model", adapter.d_model) != adapter.d_model:
            raise ValueError(f"lens d_model={state['d_model']} but model d_model={adapter.d_model}")
        return cls(adapter, state["J"], state.get("means"), state.get("n_prompts"))

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
    def unit_vectors(self, layer, token_ids, chunk=16384):
        """Unit-normalized J-lens vectors (a dictionary of atoms), computed in chunks."""
        ids = torch.as_tensor(token_ids, device=self.adapter.device)
        out = torch.empty(len(ids), self.adapter.d_model, device=self.adapter.device)
        for i in range(0, len(ids), chunk):
            v = self.vectors(layer, ids[i:i + chunk])
            out[i:i + chunk] = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return out

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

    def center(self, h, layer):
        return h.float() - self.means[layer] if layer in self.means else h.float()

    def cosine(self, h, layer, word):
        """Per-concept probe: cos(v_word, h), comparable across layers and tokens."""
        v = self.vector(layer, word)
        return torch.nn.functional.cosine_similarity(h.float(), v.expand_as(h.float()), dim=-1)

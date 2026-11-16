"""Causal interventions in J-lens coordinates.

Steering / injection     h <- h + alpha * v_tok
Directional ablation     h <- h - proj_{v_tok}(h)
Coordinate swap          V = [v_s v_t],  c = V^+ h,  h <- h + alpha * V (sigma(c) - c)
                         sigma swaps the two coordinates; the component of h
                         orthogonal to span{v_s, v_t} is left untouched.
J-space ablation         at each position, zero the projections onto the top-k
                         most active J-lens vectors (optionally exempting tokens).

All functions return context managers; interventions are active only inside
the `with` block. `positions` restricts the edit to a slice / index tensor of
sequence positions (default: every position, including generated ones).
"""
from contextlib import ExitStack, contextmanager

import torch


def _apply(h, positions, fn):
    if positions is None:
        return fn(h)
    h = h.clone()
    h[:, positions] = fn(h[:, positions])
    return h


def _project_out(h, V):
    """Remove the component of h in span(rows of V). h: (..., d), V: (m, d)."""
    P = torch.linalg.pinv(V.T) @ h.unsqueeze(-1)            # (..., m, 1) coordinates
    return h - (V.T @ P).squeeze(-1)


@contextmanager
def _over_layers(adapter, layers, make_fn):
    with ExitStack() as stack:
        for l in ([layers] if isinstance(layers, int) else layers):
            stack.enter_context(adapter.patch(l, make_fn(l)))
        yield


def steer(lens, layers, word, alpha, positions=None):
    """Inject the J-lens vector of `word` (alpha < 0 suppresses it)."""
    def make(l):
        v = lens.vector(l, word)
        return lambda h: _apply(h, positions, lambda x: x + (alpha * v).to(x.dtype))
    return _over_layers(lens.adapter, layers, make)


def ablate(lens, layers, words, positions=None):
    """Project out the J-lens vectors of `words`."""
    ids = [lens.adapter.token_id(w) for w in words]

    def make(l):
        V = lens.vectors(l, ids)
        return lambda h: _apply(h, positions, lambda x: _project_out(x.float(), V).to(x.dtype))
    return _over_layers(lens.adapter, layers, make)


def swap(lens, layers, source, target, alpha=1.0, positions=None):
    """Exchange the J-lens coordinates of `source` and `target` (e.g. 'France' <-> 'China')."""
    ids = [lens.adapter.token_id(source), lens.adapter.token_id(target)]

    def make(l):
        V = lens.vectors(l, ids).T                            # (d, 2)
        V_pinv = torch.linalg.pinv(V)                          # (2, d)

        def fn(x):
            xf = x.float()
            c = xf @ V_pinv.T                                  # (..., 2)
            delta = (c.flip(-1) - c) @ V.T                     # V (sigma(c) - c)
            return (xf + alpha * delta).to(x.dtype)
        return lambda h: _apply(h, positions, fn)
    return _over_layers(lens.adapter, layers, make)


def ablate_jspace(lens, layers, k=10, exempt=(), positions=None):
    """Zero the projections onto the k most active J-lens vectors at every position.

    `exempt` token ids are never ablated (the paper exempts tokens in the clean
    top-10 output so that the answer itself is not deleted).
    """
    exempt = torch.as_tensor(list(exempt), dtype=torch.long)

    def make(l):
        def fn(x):
            xf = x.float()
            act = lens.scores(xf, l)                           # (B, T, V) = <v_tok, h>
            if exempt.numel():
                act[..., exempt.to(act.device)] = -float("inf")
            top = act.topk(k, dim=-1).indices                  # (B, T, k)
            out = torch.empty_like(xf)
            for b in range(xf.shape[0]):
                for t in range(xf.shape[1]):
                    out[b, t] = _project_out(xf[b, t], lens.vectors(l, top[b, t]))
            return out.to(x.dtype)
        return lambda h: _apply(h, positions, fn)
    return _over_layers(lens.adapter, layers, make)

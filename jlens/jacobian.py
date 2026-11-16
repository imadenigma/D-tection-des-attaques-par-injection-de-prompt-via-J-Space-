"""Estimation of the averaged Jacobians that define the J-lens.

    J_l = E_{prompt, t, t' >= t} [ d h_{final,t'} / d h_{l,t} ]      (d_model x d_model)

J_l is the average linearized effect of a layer-l activation on the final
residual stream at the same or any later position, i.e. on what the model may
say now or later. Averaging over contexts keeps what is "poised to be spoken
about" and washes out what was only verbalized in one particular prompt.

Estimator (same reduction as Anthropic's reference implementation)
------------------------------------------------------------------
For one prompt, one forward pass is run on the prompt replicated `row_batch`
times along the batch axis and the graph is kept. Batch copy b then receives a
one-hot cotangent on output dimension i_b at *every* valid target position, and
a single backward pass gives, at every source position t,

    g_t[i_b, :] = sum_{t' >= t, t' valid} d h_{final,t'}[i_b] / d h_{l,t}

for every requested layer at once (causal masking makes t' >= t automatic).
Row i_b of the per-prompt J_l is the mean of g_t over valid source positions.
`ceil(d_model / row_batch)` backward passes reuse the same forward graph.
Averaging over prompts gives J_l.

Valid positions: the first `skip_first` positions (attention sinks, atypical
norms) and the last position (no next-token target) are excluded both as
sources and as targets.

Because the lens is read through the final normalization, the overall scale of
J_l does not change readouts; it only matters when J-lens vectors are used as
steering directions, which we always normalize.
"""
import math

import torch

SKIP_FIRST = 16


def valid_positions(T, skip_first=SKIP_FIRST):
    idx = torch.arange(skip_first, T - 1)
    if idx.numel() == 0:
        raise ValueError(f"prompt too short: {T} tokens, need > {skip_first + 1}")
    return idx


def jacobians_for_prompt(adapter, prompt, layers, target_layer=None, row_batch=16, max_len=128,
                         skip_first=SKIP_FIRST):
    """Per-prompt J_l for every l in `layers`. Returns ({l: (d, d) float32 cpu}, n_tokens)."""
    d = adapter.d_model
    target = adapter.final_layer if target_layer is None else target_layer
    layers = sorted(set(layers))
    if layers[-1] >= target:
        raise ValueError("source layers must be below the target layer")

    ids = adapter.tokenize(prompt, max_len)
    T = ids.shape[1]
    pos = valid_positions(T, skip_first).to(adapter.device)
    J = {l: torch.zeros(d, d, dtype=torch.float32) for l in layers}
    n_passes = math.ceil(d / row_batch)

    with torch.enable_grad(), adapter.capture(layers + [target], grad=True) as acts:
        adapter.model(ids.expand(row_batch, T))
        out = acts[target]                                    # (B, T, d)
        srcs = [acts[l] for l in layers]
        b_idx = torch.arange(row_batch, device=out.device)
        cot = torch.zeros_like(out)
        for p, r0 in enumerate(range(0, d, row_batch)):
            n = min(row_batch, d - r0)
            cot.zero_()
            cot[b_idx[:n, None], pos[None, :], (r0 + b_idx[:n])[:, None]] = 1.0
            grads = torch.autograd.grad(out, srcs, grad_outputs=cot, retain_graph=p < n_passes - 1)
            for l, g in zip(layers, grads):
                J[l][r0:r0 + n] = g[:n, pos].float().mean(dim=1).cpu()
    return J, T


def estimate_jacobians(adapter, prompts, layers, target_layer=None, row_batch=16, max_len=128,
                       skip_first=SKIP_FIRST, verbose=True):
    """Average per-prompt Jacobians over `prompts`. Returns ({layer: J_l}, n_prompts_used)."""
    layers = sorted(set(layers))
    total, n = None, 0
    for i, prompt in enumerate(prompts):
        try:
            J, T = jacobians_for_prompt(adapter, prompt, layers, target_layer, row_batch, max_len, skip_first)
        except ValueError as exc:
            if verbose:
                print(f"[jacobian] skip prompt {i}: {exc}")
            continue
        if total is None:
            total = J
        else:
            for l in layers:
                total[l] += J[l]
        n += 1
        if verbose:
            print(f"[jacobian] prompt {i + 1}/{len(prompts)}  T={T}", flush=True)
    if n == 0:
        raise ValueError("No usable prompts")
    return {l: total[l] / n for l in layers}, n


@torch.no_grad()
def estimate_means(adapter, prompts, layers, max_len=128, skip_first=SKIP_FIRST):
    """Mean residual activation per layer over the same kind of corpus (used for centering)."""
    layers = sorted(set(layers))
    s = {l: torch.zeros(adapter.d_model, dtype=torch.float64) for l in layers}
    n = 0
    for prompt in prompts:
        ids = adapter.tokenize(prompt, max_len)
        if ids.shape[1] <= skip_first + 1:
            continue
        acts = adapter.activations(ids, layers)
        for l in layers:
            x = acts[l][0, skip_first:].double().cpu()
            s[l] += x.sum(0)
        n += ids.shape[1] - skip_first
    return {l: (s[l] / max(n, 1)).float() for l in layers}

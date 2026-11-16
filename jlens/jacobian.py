"""Estimation of the averaged Jacobians that define the J-lens.

    J_l = E_{prompt, t, t' >= t} [ d h_{final,t'} / d h_{l,t} ]      (d_model x d_model)

J_l is the average linearized effect of a layer-l activation on the final
residual stream at the same or any later position, i.e. on what the model may
say now or later. Averaging over contexts keeps what is "poised to be spoken
about" and washes out what was only verbalized in one particular prompt.

Estimator
---------
Row i of sum_{t, t'>=t} d h_{final,t'} / d h_{l,t} is obtained with a single
vector-Jacobian product: back-propagate the scalar  sum_{t'} h_{final,t'}[i]
and sum the resulting gradient over source positions t. Causal masking makes
the t' >= t constraint automatic. Rows are computed `row_batch` at a time by
replicating the prompt along the batch axis (one basis direction per copy),
and every requested layer is read from the same backward pass.
Dividing by the number of (t, t') pairs gives the per-prompt mean; averaging
over prompts gives J_l.
"""
import torch


def _n_pairs(T, start):
    # number of (t, t') with start <= t <= t' < T
    n = T - start
    return n * (n + 1) / 2


def estimate_jacobians(adapter, prompts, layers, row_batch=16, max_len=128, skip_first=True, verbose=True):
    """Return ({layer: J_l}, {layer: mean activation}) averaged over `prompts`.

    skip_first: exclude position 0 as a source position (the BOS/attention-sink
    token has atypical norm and dominates the average otherwise).
    """
    d, L = adapter.d_model, adapter.final_layer
    layers = sorted(set(layers))
    J = {l: torch.zeros(d, d, dtype=torch.float32, device=adapter.device) for l in layers}
    mu = {l: torch.zeros(d, dtype=torch.float32, device=adapter.device) for l in layers}
    start = 1 if skip_first else 0
    n_prompts = 0

    for p_idx, prompt in enumerate(prompts):
        ids = adapter.tokenize(prompt, max_len)
        T = ids.shape[1]
        if T <= start:
            continue
        pairs = _n_pairs(T, start)

        for r0 in range(0, d, row_batch):
            rows = torch.arange(r0, min(r0 + row_batch, d), device=adapter.device)
            B = rows.numel()
            with adapter.capture(layers + [L], grad=True) as acts:
                adapter.model(ids.expand(B, T))
            h_final = acts[L]  # (B, T, d)
            # copy b back-propagates sum_{t'} h_final[b, t', rows[b]]
            target = h_final[torch.arange(B, device=adapter.device)[:, None], :, rows[:, None]].sum()
            grads = torch.autograd.grad(target, [acts[l] for l in layers])
            for l, g in zip(layers, grads):
                J[l][rows] += g[:, start:].sum(dim=1).float() / pairs
            if r0 == 0:
                for l in layers:
                    mu[l] += acts[l][0, start:].detach().float().mean(dim=0)
            del acts, grads, h_final, target

        n_prompts += 1
        if verbose:
            print(f"[jacobian] prompt {p_idx + 1}/{len(prompts)}  T={T}")

    if n_prompts == 0:
        raise ValueError("No usable prompts")
    return {l: J[l] / n_prompts for l in layers}, {l: mu[l] / n_prompts for l in layers}

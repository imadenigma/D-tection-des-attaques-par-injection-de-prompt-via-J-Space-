# Notes: J-lens, J-space and prompt-injection detection

This project turns the theory from Anthropic's paper *"Verbalizable Representations
Form a Global Workspace in Language Models"* (Transformer Circuits, July 2026) into
code that runs on open HuggingFace causal LMs, and applies it to detecting
prompt-injection attacks.

- Paper: <https://transformer-circuits.pub/2026/workspace/index.html>
- arXiv: <https://arxiv.org/abs/2607.15495>

> The paper's experiments were run on Claude (Sonnet 4.5, with checks on Haiku 4.5 and
> Opus 4.5). Everything here is a reimplementation of the published definitions on open
> models. No results from the paper are reproduced or guaranteed.

---

## 1. Background: the idea of a global workspace

In cognitive science, *global workspace theory* says the brain has many specialized
processes working in parallel, plus a small shared "stage" whose contents are broadcast
to everything else. Contents on that stage are what you can **report**, **reason with**
and **use flexibly**.

The paper asks whether LLMs have something similar. Its strategy is to first find
representations that are **verbalizable** (the model is ready to say them), and then
check whether those representations also play the other workspace roles:

| Role | What the paper checks |
|---|---|
| Report | The model can say what is represented (lens readouts, injected-thought self-reports) |
| Directed modulation | Instructions change what is active ("think about X") |
| Internal reasoning | Intermediate steps of multi-hop reasoning show up and are causally used |
| Flexible generalization | Swapping a concept changes downstream answers coherently |
| Selectivity / capacity | Only a small number of concepts (~25) are active at once |

The space of such representations is called the **J-space**, and the tool used to find
it is the **J-lens** (Jacobian lens).

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `h_{l,t}` | Residual stream after transformer block `l` at token position `t` (shape `d_model`) |
| `h_{final,t}` | Residual stream after the last block, **before** the final norm |
| `norm` | The model's final normalization layer (LayerNorm / RMSNorm) |
| `W_U` | Unembedding matrix, shape `(n_vocab, d_model)` |
| `J_l` | Averaged Jacobian from layer `l` to the final layer, shape `(d_model, d_model)` |
| `v_tok` | J-lens vector of token `tok` at layer `l` |

---

## 3. From the logit lens to the Jacobian lens

### 3.1 Logit lens (baseline)

    LogitLens(h_l) = W_U · norm(h_l)

Decode an intermediate activation as if it were already the last layer. It works in
late layers but gives noise in middle layers, because each layer represents things in a
different "coordinate system" from the output.

### 3.2 The averaged Jacobian

    J_l = E_{prompt, t, t' ≥ t} [ ∂h_{final,t'} / ∂h_{l,t} ]

- `∂h_{final,t'} / ∂h_{l,t}` is the linearized effect of a change at layer `l`, position
  `t`, on the final residual stream at position `t'`.
- `t' ≥ t` includes the **same and every later** position. So `J_l` captures what the
  model may say **now or later**, not only the next token.
- The expectation is over many prompts (the paper uses ~1,000 pretraining-like prompts).
  Averaging keeps what is generally "poised to be spoken about" and washes out what is
  specific to one context.

### 3.3 Jacobian lens

    JLens(h_l) = W_U · norm(J_l · h_l)

`J_l` carries the layer-`l` activation into final-layer coordinates, then it is decoded
as usual. This is the logit lens with `J_l` inserted, and it gives readable outputs in
middle layers.

### 3.4 J-lens vectors

For each token `tok`, the J-lens vector is the corresponding row of `W_U · J_l`:

    v_tok = J_lᵀ · W_U[tok]          (a direction in layer-l residual space)

The pre-norm lens logit of `tok` is just the inner product `⟨v_tok, h_l⟩`. So each
vocabulary token gets a direction meaning "potential to verbalize this token". There are
`n_vocab` of them in `d_model` dimensions: the set is **overcomplete** and the vectors
are **not orthogonal**.

---

## 4. J-space

### 4.1 Definition

    J-space_k = { Σ_{i∈S} c_i · v_i  :  |S| ≤ k,  c_i ≥ 0 }

The set of activations that can be written as a **sparse, nonnegative** combination of
at most `k` J-lens vectors. It is a union of many low-dimensional cones, **not** a
vector subspace.

### 4.2 Decomposition of an activation

    h = h_J + h_⊥,      h_J = argmin_{x ∈ J-space_k} ‖h − x‖

- `h_J` is the J-space component (what is in the workspace).
- `h_⊥` is everything else (the residual).
- The coefficients `c_i` are the local **J-space coordinates**.

Finding the exact closest point is hard (combinatorial), so the paper approximates it
with **gradient pursuit**.

### 4.3 How many atoms (choosing `k`)

Atoms are added greedily until the next atom improves the reconstruction **less than a
random direction of the same size would**. In the paper's middle layers, the median
occupancy plateaus around **25** concepts, and at that point the J-space explains only
about **10% of the activation variance** above the random control. The workspace is
small; most of the activation is something else.

The `k` used in the paper varies by experiment: ~25 for occupancy and probes, 16 for
concept vectors, 10 for ablations.

### 4.4 Where the workspace lives

On a depth scale reindexed 0–100, the paper places the workspace roughly at layers
**38 to 92**. Early layers are still building representations, the very last layers are
committed to the output token.

---

## 5. Interventions (causal tests)

| Intervention | Formula | Paper example |
|---|---|---|
| Steering / injection | `h ← h + α · v_tok` | Inject "lightning" into the user turn and ask the model if it notices an injected thought |
| Directional ablation | remove the projection of `h` on `v_tok` | Suppress one concept |
| Coordinate swap | `V = [v_s v_t]`, `c = V⁺h`, `h ← h + α · V(σ(c) − c)` | France→China, spider→ant (8 legs becomes 6), fight→light |
| J-space ablation | at each position, zero projections onto the top-k (k=10) active J-lens vectors, exempting the clean top-10 output tokens | Multi-hop reasoning collapses while routine tasks survive |

- `V⁺` is the pseudoinverse; it is needed because J-lens vectors are not orthogonal.
- `σ` swaps the two coordinates. The part of `h` orthogonal to `span{v_s, v_t}` is
  untouched.
- Reported effect sizes are modest: an ordinary-strength France→China swap gave the
  target-appropriate answer in 76/192 trials, 101/192 at double strength.

---

## 6. Use for safety monitoring and prompt injection

The paper uses J-space readouts for auditing:

- surfacing evaluation awareness (concepts like "fake", "fictional");
- exposing hidden motives in deliberately misaligned models ("reward", "bias",
  "secretly", "trick");
- one illustrative figure where the lens flags a suspicious search result as a
  **prompt injection** while the model reads it.

The paper does **not** provide an evaluated detector, threshold or error rates. The
monitor in this project is our own construction on top of those ideas.

### 6.1 Threat model

An LLM agent receives a trusted instruction plus **untrusted data** (tool output,
retrieved page, email). An attacker hides instructions in that data ("ignore previous
instructions and …"). Hypothesis: while reading such text, injection-related concepts
become active in the workspace, even if the model never writes them out.

### 6.2 Scoring (`jlens/monitor.py`)

For each token `t` inside the untrusted span and each workspace layer `l`:

1. **Probe signal**: for each concept word `w` in a lexicon (`injection`, `ignore`,
   `override`, `suspicious`, `trick`, `secretly`, …):

       probe_w = cos(v_w, h_{l,t})

   averaged over layers; the token's probe score is the max over concepts.

2. **J-space signal**: decompose `h_{l,t}` (mean-centered) into J-space and take the
   share of coefficient mass that falls on concept atoms:

       jspace = Σ_{i ∈ S ∩ concepts} c_i / Σ_{i ∈ S} c_i

   averaged over layers.

3. **Token score** = `w · probe + (1 − w) · jspace` (default `w = 0.5`).

4. **Span score** = max over tokens. Flag if `score ≥ threshold`.

The report also lists, per token, the top concepts and the full workspace content
(averaged J-space coordinates), so a human can see *why* a span was flagged.

### 6.3 Limits

- No threshold is shipped. Calibrate it on your own benign vs. injected examples.
- The concept lexicon is hand-written. Multi-token words are represented by their
  first token only.
- The probe signal can fire on benign text that merely talks about "attacks" or
  "instructions" (e.g. a security tutorial). The J-space signal and the per-token
  workspace listing help, but do not remove that problem.
- The lens is fitted on a tiny built-in corpus by default; quality improves with more
  prompts.

---

## 7. Code map

```
main.py                 CLI: fit / read / decompose / steer / swap / monitor / train-detector / detect / defend
jlens/
  model.py              ModelAdapter: hooks for h_l, final norm, W_U, chat template, patching
  jacobian.py           J_l estimator (reference reduction) and mean activations
  lens.py               JacobianLens: readouts, J-lens vectors, load/save (incl. published lenses)
  jspace.py             JSpace (single activation) and batched_pursuit (many activations)
  features.py           Featurizer: J-space, logit-lens and raw features of a prompt
  detector.py           JSpaceDetector: logistic regression on J-space coordinates
  defense.py            SteeringDefense: detector-gated steering, refusal judge
  interventions.py      steer, ablate, swap, ablate_jspace (context managers)
  monitor.py            InjectionMonitor (lexicon-based, token level)
experiments/            00-06: data, lens validation, workspace band, features, classification, defense, figures
```

### 7.1 `model.py`

- Finds decoder blocks, final norm and `W_U` for common architectures (Llama, Qwen 2/3/3.5,
  Mistral, Gemma, GPT-2, GPT-NeoX, OPT).
- `capture(layers, grad=False)` records `h_l` during a forward pass; with `grad=True` the
  embedding output is a gradient leaf so captured activations stay in the autograd graph.
- `patch(layer, fn)` replaces `h_l` by `fn(h_l)` (used by every intervention).
- `chat(user, system=None)` renders the chat template (thinking disabled on Qwen3/3.5).

### 7.2 `jacobian.py` – how `J_l` is computed

We use exactly the reduction of Anthropic's reference implementation, so that our lenses
are comparable with the published ones:

- One forward pass on the prompt replicated `row_batch` times; the graph is kept.
- Copy `b` receives a one-hot cotangent on output dimension `i_b` **at every valid target
  position at once**. One backward pass gives, at every source position `t`,
  `Σ_{t' ≥ t} ∂h_{final,t'}[i_b] / ∂h_{l,t}` (causal attention enforces `t' ≥ t`), for every
  requested layer.
- Row `i_b` of the per-prompt `J_l` is the mean over valid source positions;
  `ceil(d_model / row_batch)` backward passes reuse the same forward graph.
- Valid positions exclude the first 16 tokens (attention sinks) and the last one.
- `J_l` is averaged over prompts. Its overall scale does not affect readouts (they go
  through the final norm).

Validation (`experiments/01_validate_lens.py`): fitted on Pythia-70m, our `J_l` matches the
lens Neuronpedia fitted with the reference code (see the paper for the numbers).

### 7.3 `lens.py`

- `logits(h, l)` = `W_U · norm(J_l h)`; `logit_lens(h)` for comparison;
  `JacobianLens.identity(...)` is the logit lens written as `J_l = I` (used as an ablation).
- `vectors(l, ids)` / `unit_vectors(l, ids)`: J-lens vectors `W_U[ids] · J_l`.
- `load(path)` reads our files **and** the published format (`neuronpedia/jacobian-lens`).
- `fit_means(prompts)`: mean residual per layer, used to center activations.

### 7.4 `jspace.py` – nonnegative gradient pursuit

Atoms are J-lens vectors normalized to unit length. Loop up to `k` times: pick the atom most
positively correlated with the residual, then a few projected gradient steps on the
coefficients of the selected set (exact line search, coefficients kept `≥ 0`). A sample
stops when the new atom removes less residual energy than a random unit direction would
(`‖r‖²/d`). `batched_pursuit` runs this for many activations at once against a fixed
dictionary (the J-lens vectors of the ~66k word tokens with a leading space), so each
dictionary matrix is read once per step for the whole batch.

### 7.5 `features.py`, `detector.py`, `defense.py`

- **Positions.** *span* = the content to inspect (the user message, or only the untrusted
  document); *tail* = the chat-template tokens after it, where the model prepares its answer.
- **Features.** Activations are centered with the layer mean on WikiText. Per prompt:
  concept-probe cosines for ~90 concept words (harm / refusal / injection lexicons + a random
  control lexicon), per-token lexicon scores, J-space coordinates of three pooled vectors
  (mean span, mean tail, last token) at the code layers, the same in logit-lens form, and the
  raw pooled residuals.
- **Detector.** Logistic regression on the J-space coordinates summed over layers: one sparse
  vector indexed by (pool, word). Weights are words, so `explain()` lists the atoms behind a
  decision.
- **Defense.** If flagged, add `α · ‖h_l‖_avg · u_l` at the workspace layers, with `u_l` the
  normalized sum of the unit J-lens vectors of refusal words. Unflagged prompts are untouched.

### 7.6 `interventions.py`, `monitor.py`

Unchanged in spirit: `steer`, `ablate`, `swap`, `ablate_jspace` are context managers;
`InjectionMonitor` scores the tokens of an untrusted span with lexicon probes + J-space mass.

---

## 8. Usage

See the README (`main.py` commands and the `experiments/` pipeline). The default model is
`Qwen/Qwen3.5-0.8B` with the published Neuronpedia lens; set `JSPACE_MODEL`, `JSPACE_LENS`
and `JSPACE_MEANS` to use local files.

---

## 9. Differences from the paper

| Topic | Paper | Here |
|---|---|---|
| Model | Claude Sonnet 4.5 (+ Haiku/Opus checks) | Qwen3.5-0.8B (open weights, hybrid linear attention) |
| Lens | Fitted on 1,000 pretraining-like prompts | Published Neuronpedia lens (reference code, WikiText); our estimator validated on Pythia-70m |
| Dictionary | Full vocabulary | Word tokens with a leading space (66k of 248k), for speed and readability |
| Sparse solver | Gradient pursuit | Nonnegative gradient pursuit, batched, random-direction stopping rule |
| Workspace layers | ~33-90% of depth | Chosen from per-layer lens quality on Qwen (see paper) |
| Injection detection | One illustrative example | Trained detector, OOD evaluation, token localization, gated steering defense |

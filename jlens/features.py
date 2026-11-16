"""J-space features of a prompt, computed from a single forward pass.

For a user message rendered with the chat template, we look at two groups of token
positions:
    span  the tokens of the content to inspect (the whole user message, or only the
          untrusted document for indirect injections),
    tail  the chat-template tokens after the user message ("<|im_end|> ... assistant"),
          where the model has read the request and prepares its answer.

Activations are centered with the layer's mean residual on generic text (`lens.means`).
Three families of features are produced, each in a J-lens version and in a logit-lens
version (J_l = I) used as an ablation:

  probe    cos(v_w, h~) for every concept word w of the lexicons, per layer, pooled as
           [max over span, mean over span, mean over tail]          -> (L, C, 3)
  tokens   per-token lexicon scores (mean over the lexicon's words)  -> (T, L, n_lex)
  codes    J-space coordinates: nonnegative sparse decomposition (k atoms) of the pooled
           activations [mean span, mean tail, last token] over a dictionary of unit J-lens
           vectors of word tokens, for the code layers            -> sparse (layer, pool, atom, coef)

plus the raw pooled residuals (L, 3, d) for the activation-probe baseline.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .jspace import batched_pursuit, word_token_ids

LEXICONS = {
    "harm": ("harmful illegal dangerous unethical violence violent weapon weapons crime criminal hack hacking "
             "steal stealing fraud malicious attack kill murder bomb explosives drugs poison exploit scam "
             "abuse terrorism terrorist suicide harm theft illicit malware virus"),
    "refusal": "sorry cannot refuse unable apologize inappropriate unsafe warning safety ethical decline "
               "prohibited forbidden against policy",
    "injection": "injection inject ignore disregard override instructions instruction suspicious trick secretly "
                 "hidden pretend fake command instead manipulation prompt bypass",
}
POOLS = ("max_span", "mean_span", "mean_tail")
CODE_POOLS = ("mean_span", "mean_tail", "last")


@dataclass
class PromptFeatures:
    n_tokens: int
    span: np.ndarray            # token indices of the span
    tail: np.ndarray            # token indices of the tail
    raw: np.ndarray             # (L, 3, d) float16: mean span, mean tail, last
    probe: dict                 # {"jlens"|"logit": (L, C, 3) float32}
    tokens: dict                # {"jlens"|"logit": (T, L, n_lex) float16}
    codes: dict                 # {"jlens"|"logit": (n, 4) float32 rows (layer_idx, pool, atom, coef)}
    next_token_top: np.ndarray  # model's top-5 next tokens after the prompt
    _pooled: object = None      # (L, 3, d) float32 tensor, dropped after add_codes


class Featurizer:
    def __init__(self, adapter, lens, logit_lens, code_layers, k=25, inner_steps=3, random_lexicon_size=25,
                 seed=0):
        self.adapter, self.lens, self.logit = adapter, lens, logit_lens
        self.layers = lens.layers
        self.code_layers = list(code_layers)
        self.k, self.inner_steps = k, inner_steps
        tok = adapter.tokenizer

        # concept words (single tokens only) and lexicon membership
        words, lex_of = [], []
        for name, ws in LEXICONS.items():
            for w in ws.split():
                t = adapter.single_token(" " + w)
                if t is not None and (" " + w) not in words:
                    words.append(" " + w)
                    lex_of.append(name)
        self.dict_ids = word_token_ids(tok)
        rng = np.random.default_rng(seed)
        concept_ids = {adapter.single_token(w) for w in words}
        pool = [i for i in self.dict_ids if i not in concept_ids]
        for i in rng.choice(len(pool), random_lexicon_size, replace=False):
            words.append(tok.decode([pool[i]]))
            lex_of.append("random")
        self.words = words
        self.word_ids = [adapter.token_id(w) for w in words]
        self.lex_names = list(LEXICONS) + ["random"]
        self.lex_mask = torch.tensor([[lx == n for lx in lex_of] for n in self.lex_names],
                                     dtype=torch.float32, device=adapter.device)      # (n_lex, C)
        self.lex_mask /= self.lex_mask.sum(1, keepdim=True)

        self.unit = {name: {l: ln.unit_vectors(l, self.word_ids) for l in self.layers}
                     for name, ln in (("jlens", lens), ("logit", logit_lens))}
        self.dicts = {"jlens": {l: lens.unit_vectors(l, self.dict_ids) for l in self.code_layers}}
        logit_dict = logit_lens.unit_vectors(self.code_layers[0], self.dict_ids)
        self.dicts["logit"] = {l: logit_dict for l in self.code_layers}

    def positions(self, rendered, char_span):
        ids, offsets = self.adapter.tokenize_with_offsets(rendered)
        lo, hi = char_span
        span = [i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]
        tail = [i for i, (a, b) in enumerate(offsets) if a >= hi and b > a]
        tail = tail or [ids.shape[1] - 1]
        return ids, np.array(span), np.array(tail)

    @torch.no_grad()
    def __call__(self, user_text, span=None, system=None):
        """`span`: (start, end) char offsets inside `user_text` of the content to inspect."""
        rendered = self.adapter.chat(user_text, system=system)
        try:
            base = rendered.index(user_text)
        except ValueError:  # chat templates may strip surrounding whitespace
            base = rendered.index(user_text.strip()) - (len(user_text) - len(user_text.lstrip()))
        lo, hi = (0, len(user_text)) if span is None else span
        ids, span_pos, tail_pos = self.positions(rendered, (base + lo, base + hi))
        logits, acts = self.adapter.forward_with_activations(ids, self.layers)
        T = ids.shape[1]
        sp = torch.as_tensor(span_pos, device=self.adapter.device)
        tp = torch.as_tensor(tail_pos, device=self.adapter.device)

        H = torch.stack([self.lens.center(acts[l], l) for l in self.layers])           # (L, T, d)
        pooled = torch.stack([H[:, sp].mean(1), H[:, tp].mean(1), H[:, -1]], 1)        # (L, 3, d)
        Hn = torch.nn.functional.normalize(H, dim=-1)

        probe, tokens = {}, {}
        for name in ("jlens", "logit"):
            U = torch.stack([self.unit[name][l] for l in self.layers])                 # (L, C, d)
            cos = torch.einsum("ltd,lcd->ltc", Hn, U)                                   # (L, T, C)
            probe[name] = torch.stack([cos[:, sp].max(1).values, cos[:, sp].mean(1), cos[:, tp].mean(1)],
                                      -1).cpu().numpy()                                 # (L, C, 3)
            tokens[name] = (cos @ self.lex_mask.T).permute(1, 0, 2).cpu().numpy().astype(np.float16)

        return PromptFeatures(
            n_tokens=T, span=span_pos, tail=tail_pos,
            raw=pooled.cpu().numpy().astype(np.float16), probe=probe, tokens=tokens, codes={},
            next_token_top=logits[-1].topk(5).indices.cpu().numpy(), _pooled=pooled)

    @torch.no_grad()
    def add_codes(self, feats):
        """J-space decomposition of the pooled activations of many prompts at once (batched so
        that each dictionary matrix is read once per pursuit step for the whole batch)."""
        li = [self.layers.index(l) for l in self.code_layers]
        P = len(CODE_POOLS)
        for name in ("jlens", "logit"):
            rows = [[] for _ in feats]
            for j, l in zip(li, self.code_layers):
                X = torch.cat([f._pooled[j] for f in feats])                                  # (n*P, d)
                S, C, _ = batched_pursuit(X, self.dicts[name][l], k=self.k, inner_steps=self.inner_steps)
                S, C = S.cpu().numpy(), C.cpu().numpy()
                for r in range(S.shape[0]):
                    keep = S[r] >= 0
                    f_idx, p = divmod(r, P)
                    rows[f_idx].extend((j, p, a, c) for a, c in zip(S[r][keep], C[r][keep]))
            for f, rr in zip(feats, rows):
                f.codes[name] = np.array(rr, dtype=np.float32).reshape(-1, 4)
        for f in feats:
            f._pooled = None
        return feats

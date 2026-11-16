"""Thin adapter over a HuggingFace causal LM exposing what the J-lens needs:
the residual stream after each block, the final normalization, and W_U.

Notation (follows "Verbalizable Representations Form a Global Workspace in
Language Models", Anthropic 2026):
    h_{l,t}      residual stream after block l at position t
    h_{final,t}  residual stream after the last block (before the final norm)
    norm         the model's final normalization layer
    W_U          unembedding matrix, shape (n_vocab, d_model)
"""
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
)
_NORM_PATHS = (
    "model.norm",
    "model.language_model.norm",
    "transformer.ln_f",
    "gpt_neox.final_layer_norm",
    "model.decoder.final_layer_norm",
)


def _resolve(root, paths):
    for path in paths:
        obj = root
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"Unsupported architecture: none of {paths} found on {type(root).__name__}")


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


class ModelAdapter:
    def __init__(self, model_name, device=None, dtype=torch.float32):
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layers = _resolve(self.model, _LAYER_PATHS)
        self.final_norm = _resolve(self.model, _NORM_PATHS)
        self.W_U = self.model.get_output_embeddings().weight  # (n_vocab, d_model)

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def d_model(self):
        return self.W_U.shape[1]

    @property
    def final_layer(self):
        return self.n_layers - 1

    def workspace_layers(self, lo=0.38, hi=0.92, stride=1):
        """Layer band where the paper locates the workspace (~L38-L92 on a 0-100 depth scale)."""
        return list(range(int(lo * self.n_layers), int(hi * self.n_layers), stride))

    def tokenize(self, text, max_len=None):
        enc = self.tokenizer(text, return_tensors="pt", truncation=max_len is not None, max_length=max_len)
        return enc["input_ids"].to(self.device)

    def tokenize_with_offsets(self, text):
        enc = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        return enc["input_ids"].to(self.device), enc["offset_mapping"][0].tolist()

    def decode(self, token_id):
        return self.tokenizer.decode([int(token_id)])

    def token_id(self, word):
        """First token of `word`. Multi-token words are truncated to their first piece."""
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            raise ValueError(f"{word!r} tokenizes to nothing")
        return ids[0]

    @contextmanager
    def capture(self, layer_ids, grad=False):
        """Record h_l for each layer in `layer_ids` during the next forward pass.

        With grad=True the embedding output is made a grad-requiring leaf so the
        captured activations stay in the autograd graph (needed for Jacobians).
        """
        acts, handles = {}, []
        for l in set(layer_ids):
            def hook(_mod, _inp, out, l=l):
                acts[l] = _hidden(out)
            handles.append(self.layers[l].register_forward_hook(hook))
        if grad:
            def emb_hook(_mod, _inp, out):
                return out.detach().requires_grad_(True)
            handles.append(self.model.get_input_embeddings().register_forward_hook(emb_hook))
        try:
            yield acts
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def patch(self, layer, fn):
        """Replace h_l by fn(h_l) (h_l has shape (batch, seq, d_model)) during forward passes."""
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)
        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def activations(self, input_ids, layer_ids):
        with self.capture(layer_ids) as acts:
            self.model(input_ids)
        return {l: a.detach() for l, a in acts.items()}

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=40, **kw):
        ids = self.tokenize(prompt)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

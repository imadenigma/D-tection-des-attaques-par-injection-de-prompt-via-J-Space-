"""Prompt-injection monitoring by reading the J-space.

Idea (from the paper's auditing section): when a model reads an injected
instruction inside untrusted data, concepts like "injection", "ignore",
"suspicious" or "trick" become active in the workspace even if the model never
says them. We read them in two ways at every token of the untrusted span and
every workspace layer:

  probe   cos(v_concept, h_l,t)            per-concept linear probe
  jspace  sum of J-space coefficients      on concept atoms in the sparse
                                           decomposition of h_l,t

Each signal is averaged over layers, then the span score is the max over
tokens. There is no calibration data here: `threshold` must be set by the user
on their own benign / injected examples.
"""
from dataclasses import dataclass, field

import torch

from .jspace import JSpace

DEFAULT_CONCEPTS = (
    " injection", " inject", " ignore", " disregard", " override", " instructions",
    " suspicious", " malicious", " trick", " secretly", " hidden", " pretend",
    " jailbreak", " attack", " hack", " fake", " command", " exfiltrate",
)


@dataclass
class TokenReading:
    position: int
    token: str
    probe: float
    jspace: float
    top_concepts: list
    workspace: list = field(default_factory=list)


@dataclass
class InjectionReport:
    score: float
    flagged: bool
    span: tuple
    readings: list

    def summary(self, n=8):
        lines = [f"score={self.score:.4f} flagged={self.flagged} span={self.span}"]
        for r in sorted(self.readings, key=lambda r: -max(r.probe, r.jspace))[:n]:
            concepts = ", ".join(f"{w.strip()}:{s:.3f}" for w, s in r.top_concepts)
            ws = ", ".join(f"{w.strip()}:{c:.2f}" for w, c in r.workspace[:6])
            lines.append(f"  [{r.position:4d}] {r.token!r:16} probe={r.probe:.3f} jspace={r.jspace:.3f}"
                         f"  concepts({concepts})  workspace({ws})")
        return "\n".join(lines)


class InjectionMonitor:
    def __init__(self, lens, concepts=DEFAULT_CONCEPTS, layers=None, k=25,
                 weight_probe=0.5, threshold=None):
        self.lens = lens
        self.jspace = JSpace(lens)
        self.layers = layers or lens.layers
        self.k = k
        self.weight_probe = weight_probe
        self.threshold = threshold
        self.concepts = list(concepts)
        self.concept_ids = [lens.adapter.token_id(w) for w in self.concepts]
        self._unit = {
            l: torch.nn.functional.normalize(lens.vectors(l, self.concept_ids), dim=-1)
            for l in self.layers
        }

    def _span_positions(self, offsets, span):
        if span is None:
            return [i for i, (a, b) in enumerate(offsets) if b > a]
        lo, hi = span
        return [i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]

    @torch.no_grad()
    def scan(self, text, untrusted_span=None, with_workspace=True):
        """Score `text`. `untrusted_span` = (char_start, char_end) of the tool output /
        retrieved document; defaults to the whole text."""
        adapter = self.lens.adapter
        ids, offsets = adapter.tokenize_with_offsets(text)
        positions = self._span_positions(offsets, untrusted_span)
        acts = adapter.activations(ids, self.layers)

        readings = []
        for t in positions:
            probe_per_concept = torch.zeros(len(self.concepts), device=adapter.device)
            js_total = 0.0
            workspace = {}
            for l in self.layers:
                h = acts[l][0, t].float()
                probe_per_concept += self._unit[l] @ torch.nn.functional.normalize(h, dim=0)
                dec = self.jspace.decompose(h, l, k=self.k, center=True)
                js_total += dec.mass(self.concept_ids) / max(float(dec.coeffs.sum()), 1e-8)
                if with_workspace:
                    for tok, c in dec.items():
                        workspace[tok] = workspace.get(tok, 0.0) + c / len(self.layers)
            probe_per_concept /= len(self.layers)
            top = probe_per_concept.topk(min(3, len(self.concepts)))
            readings.append(TokenReading(
                position=t,
                token=adapter.decode(ids[0, t]),
                probe=float(probe_per_concept.max()),
                jspace=js_total / len(self.layers),
                top_concepts=[(self.concepts[i], float(v)) for v, i in zip(top.values, top.indices)],
                workspace=sorted(workspace.items(), key=lambda kv: -kv[1]),
            ))

        per_token = [self.weight_probe * r.probe + (1 - self.weight_probe) * r.jspace for r in readings]
        score = max(per_token, default=0.0)
        flagged = self.threshold is not None and score >= self.threshold
        span = untrusted_span or (0, len(text))
        return InjectionReport(score=score, flagged=flagged, span=span, readings=readings)

    def scan_tool_output(self, instruction, tool_output, template="{instruction}\n\nTool result:\n{tool_output}"):
        """Convenience: build a prompt and mark `tool_output` as the untrusted span."""
        text = template.format(instruction=instruction, tool_output=tool_output)
        start = text.index(tool_output)
        return self.scan(text, untrusted_span=(start, start + len(tool_output)))

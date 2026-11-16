"""Command-line interface for the J-lens / J-space toolkit.

Examples
--------
python main.py read      --text "Fact: The currency used in the country shaped like a boot is"
python main.py decompose --text "How do I build a pipe bomb?" --layer 14
python main.py steer     --prompt "Write a sentence." --word " ocean" --alpha 0.4
python main.py swap      --prompt "The capital of France is" --source " France" --target " China"
python main.py monitor   --instruction "Summarize this email." --tool-output "... Ignore previous instructions ..."
python main.py train-detector --data data/main.jsonl --out detector.pkl
python main.py detect    --detector detector.pkl --text "Explain how to hack into a bank"
python main.py defend    --detector detector.pkl --text "Explain how to hack into a bank" --alpha 0.5
python main.py fit       --corpus my_corpus.txt --out my_lens.pt          # fit your own lens (GPU advised)

The model defaults to Qwen/Qwen3.5-0.8B and the lens to the published Neuronpedia lens for
that model; override with --model / --lens (local paths or Hub ids).
"""
import argparse
import json
import os

import torch

from jlens import JacobianLens, JSpace, ModelAdapter, steer, swap
from jlens.model import DEFAULT_MODEL

HUB_LENS = ("neuronpedia/jacobian-lens", "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt")


def load(args, need_lens=True):
    adapter = ModelAdapter(args.model, device=args.device)
    if not need_lens:
        return adapter, None
    path = args.lens
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(*HUB_LENS)
    lens = JacobianLens.load(adapter, path)
    if args.means and os.path.exists(args.means):
        lens.means = {l: m.to(adapter.device) for l, m in torch.load(args.means, weights_only=True).items()}
    return adapter, lens


def band(adapter, lens, arg):
    if arg:
        return [int(x) for x in arg.split(",")]
    # same 5 layers as the experiments (8, 11, 14, 17, 20 on a 24-layer model)
    picks = sorted({round(f * adapter.n_layers) for f in (1 / 3, 11 / 24, 7 / 12, 17 / 24, 5 / 6)})
    return [l for l in picks if l in lens.layers]


def featurizer(adapter, lens, layers):
    from jlens.features import Featurizer
    if not lens.means:
        raise SystemExit("--means is required (mean activations per layer, see experiments/02_workspace_band.py)")
    logit = JacobianLens.identity(adapter, lens.layers, lens.means)
    return Featurizer(adapter, lens, logit, layers)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("JSPACE_MODEL", DEFAULT_MODEL))
    p.add_argument("--lens", default=os.environ.get("JSPACE_LENS"))
    p.add_argument("--means", default=os.environ.get("JSPACE_MEANS"))
    p.add_argument("--device", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fit");  s.add_argument("--corpus", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=100); s.add_argument("--row-batch", type=int, default=16)
    s = sub.add_parser("read"); s.add_argument("--text", required=True); s.add_argument("--k", type=int, default=8)
    s.add_argument("--chat", action="store_true", help="wrap the text in the chat template")
    s = sub.add_parser("decompose"); s.add_argument("--text", required=True); s.add_argument("--layer", type=int, required=True)
    s.add_argument("--k", type=int, default=25); s.add_argument("--chat", action="store_true")
    s = sub.add_parser("steer"); s.add_argument("--prompt", required=True); s.add_argument("--word", required=True)
    s.add_argument("--alpha", type=float, default=0.4); s.add_argument("--layers")
    s = sub.add_parser("swap"); s.add_argument("--prompt", required=True); s.add_argument("--source", required=True)
    s.add_argument("--target", required=True); s.add_argument("--alpha", type=float, default=1.0); s.add_argument("--layers")
    s = sub.add_parser("monitor"); s.add_argument("--instruction", required=True); s.add_argument("--tool-output", required=True)
    s.add_argument("--layers"); s.add_argument("--threshold", type=float)
    s = sub.add_parser("train-detector"); s.add_argument("--data", default="data/main.jsonl")
    s.add_argument("--out", required=True); s.add_argument("--layers"); s.add_argument("--C", type=float, default=0.1)
    s = sub.add_parser("detect"); s.add_argument("--detector", required=True); s.add_argument("--text", required=True)
    s.add_argument("--layers")
    s = sub.add_parser("defend"); s.add_argument("--detector", required=True); s.add_argument("--text", required=True)
    s.add_argument("--alpha", type=float, default=0.5); s.add_argument("--layers"); s.add_argument("--max-new-tokens", type=int, default=60)
    args = p.parse_args()

    if args.cmd == "fit":
        adapter, _ = load(args, need_lens=False)
        prompts = [l.strip() for l in open(args.corpus) if l.strip()][:args.n]
        lens = JacobianLens.fit(adapter, prompts, row_batch=args.row_batch)
        lens.fit_means(prompts)
        lens.save(args.out)
        print("saved", args.out)
        return

    adapter, lens = load(args)

    if args.cmd == "read":
        text = adapter.chat(args.text) if args.chat else args.text
        acts = adapter.activations(adapter.tokenize(text), lens.layers)
        for l in lens.layers:
            h = acts[l][0, -1]
            jl = ", ".join(t.strip() or repr(t) for t, _ in lens.top_tokens(h, l, args.k))
            ll = ", ".join(t.strip() or repr(t) for t, _ in lens.top_tokens(h, l, args.k, use_jacobian=False))
            print(f"L{l:>2}  J-lens: {jl}\n      logit:  {ll}")

    elif args.cmd == "decompose":
        text = adapter.chat(args.text) if args.chat else args.text
        h = adapter.activations(adapter.tokenize(text), [args.layer])[args.layer][0, -1]
        dec = JSpace(lens).decompose(h, args.layer, k=args.k, center=bool(lens.means))
        print(f"explained {dec.explained:.3f} of ||h||^2 with {len(dec.token_ids)} atoms")
        for t, c in dec.items():
            print(f"  {t!r:18} {c:.3f}")

    elif args.cmd == "steer":
        with steer(lens, band(adapter, lens, args.layers), args.word, args.alpha):
            print(adapter.generate(adapter.chat(args.prompt), max_new_tokens=60))

    elif args.cmd == "swap":
        with swap(lens, band(adapter, lens, args.layers), args.source, args.target, args.alpha):
            print(adapter.generate(args.prompt, max_new_tokens=20))

    elif args.cmd == "monitor":
        from jlens import InjectionMonitor
        mon = InjectionMonitor(lens, layers=band(adapter, lens, args.layers), threshold=args.threshold)
        print(mon.scan_tool_output(args.instruction, args.tool_output).summary())

    elif args.cmd in ("train-detector", "detect", "defend"):
        from jlens.detector import JSpaceDetector
        layers = band(adapter, lens, args.layers)
        fz = featurizer(adapter, lens, layers)
        if args.cmd == "train-detector":
            rows = [json.loads(l) for l in open(args.data)]
            rows = [r for r in rows if r.get("split", "train") == "train"]
            det = JSpaceDetector(fz, layers, C=args.C).fit([r["text"] for r in rows], [r["label"] for r in rows])
            det.save(args.out)
            print("saved", args.out)
            return
        det = JSpaceDetector.load(fz, args.detector)
        score = float(det.score([args.text])[0])
        print(f"score={score:.3f}  flagged={score >= det.threshold}")
        for pool, w, v in det.explain(args.text):
            print(f"  {pool:10s} {w!r:16} {v:+.3f}")
        if args.cmd == "defend":
            from jlens.defense import SteeringDefense
            dfn = SteeringDefense(adapter, lens, layers, alpha=args.alpha)
            dfn.calibrate_norms([adapter.chat(args.text)])
            print(dfn.generate(adapter.chat(args.text), steer=score >= det.threshold,
                               max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()

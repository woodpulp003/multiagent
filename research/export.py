"""Export a trained denoiser to submission/weights/denoiser.onnx + meta.json.

The bot loads ONNX with onnxruntime because importing PyTorch could exceed the
15-second initialization limit. The export is checked numerically before use.

    python -m research.export --checkpoint checkpoints/denoiser.pt
    python -m research.export --untrained     # random weights, plumbing tests only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import CHECKPOINTS, WEIGHTS
import features
from .model import Denoiser

NAMES = ["context", "tokens", "t"]


def example_inputs(batch: int):
    tokens = torch.randint(0, features.NUM_TOKENS, (batch, 64))
    return torch.randn(batch, features.CONTEXT_PLANES, 8, 8), tokens, torch.rand(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "denoiser.pt")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--out", type=Path, default=WEIGHTS)
    parser.add_argument("--steps", type=int, default=8, help="reverse-chain length used by the bot")
    args = parser.parse_args()
    torch.manual_seed(7980)
    if args.untrained:
        model = Denoiser()
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != features.FORMAT_VERSION:
            raise SystemExit("Checkpoint was trained with a different features.FORMAT_VERSION")
        model = Denoiser(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "denoiser.onnx"
    torch.onnx.export(model, example_inputs(1), str(onnx_path), input_names=NAMES,
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={name: {0: "batch"} for name in NAMES + ["logits"]})
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    for batch in (1, 32):
        inputs = example_inputs(batch)
        with torch.inference_mode():
            expected = model(*inputs).numpy()
        actual = session.run(None, {n: v.numpy() for n, v in zip(NAMES, inputs)})[0]
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-4)
    meta = {"format_version": features.FORMAT_VERSION, "trained": not args.untrained,
            "steps": args.steps, "config": model.config,
            "parameters": sum(p.numel() for p in model.parameters())}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) {meta}")


if __name__ == "__main__":
    main()

"""Train the denoiser with the masked denoising cross-entropy (note, section 3.1).

For each example: sample a masking level t ~ U(0, 1], hide each hidden square's
true label with probability t, and predict the clean label of masked squares.
Train/validation split is by whole game, never by individual turns.

    python -m research.train --data data/selfplay --epochs 5

SCAFFOLD: works end to end; tune batch size, schedule, loss weighting, etc.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import CHECKPOINTS, DATA
import features
from .model import Denoiser


TENSOR_KEYS = ("context", "tokens", "labels", "game_id")


def load_shards(directory: Path):
    shards = [np.load(path) for path in sorted(directory.glob("**/*.npz"))]
    if not shards:
        raise SystemExit(f"No .npz shards in {directory}; run research.selfplay first")
    return {key: np.concatenate([shard[key] for shard in shards]) for key in TENSOR_KEYS}


def split_by_game(game_ids: np.ndarray, validation_fraction: float, seed: int):
    games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(games, max(1, int(len(games) * validation_fraction)), replace=False))
    is_validation = np.array([game in held_out for game in game_ids])
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


def corrupt(tokens: torch.Tensor, labels: torch.Tensor, t: torch.Tensor):
    """Forward process q: reveal true labels on hidden squares, then re-mask with prob t."""
    hidden = labels != features.IGNORE_LABEL
    masked = hidden & (torch.rand(labels.shape) < t[:, None])
    noisy = torch.where(hidden, labels.clamp(min=0), tokens)
    noisy = torch.where(masked, torch.full_like(noisy, features.MASK), noisy)
    return noisy, masked


def batch_loss(model, context, tokens, labels):
    t = torch.rand(labels.shape[0]).clamp(min=1e-3)
    noisy, masked = corrupt(tokens, labels, t)
    logits = model(context, noisy, t)
    targets = torch.where(masked, labels, torch.full_like(labels, features.IGNORE_LABEL))
    return F.cross_entropy(logits.reshape(-1, features.NUM_CLASSES), targets.reshape(-1),
                           ignore_index=features.IGNORE_LABEL)


def tensors(data, index):
    context = data["context"][index].astype(np.float32)
    if data["context"].dtype == np.uint8:
        context /= 255.0
    return (torch.from_numpy(context),
            torch.from_numpy(data["tokens"][index].astype(np.int64)),
            torch.from_numpy(data["labels"][index].astype(np.int64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "round0" / "train")
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "denoiser.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = load_shards(args.data)
    train_index, val_index = split_by_game(data["game_id"], 0.1, args.seed)
    print(f"train={len(train_index)} val={len(val_index)} examples")
    model = Denoiser(args.channels, args.blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(train_index)
        for start in range(0, len(order), args.batch_size):
            loss = batch_loss(model, *tensors(data, order[start:start + args.batch_size]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = [batch_loss(model, *tensors(data, val_index[s:s + 1024])).item()
                      for s in range(0, len(val_index), 1024)]
        print(f"epoch={epoch} train_loss={loss.item():.4f} val_loss={np.mean(losses):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "format_version": features.FORMAT_VERSION}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

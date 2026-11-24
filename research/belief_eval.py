"""Score belief samplers against the true hidden board (note, section 5).

Plays fresh games against the held-out opponent pool. At randomly chosen
turns, every belief draws K boards from the mover's tracker and is scored
against the truth with the same sample-frequency estimator:

  square_nll    mean -log p(true class) over hidden squares (Laplace-smoothed)
  king_nll      -log p(true enemy-king square) when the king is hidden
  king_top1     most frequent sampled king square is correct
  ploss_brier   mean (p_loss_hat(a) - L(true, a))^2 over legal moves
  loss_recall   fraction of truly king-losing moves with p_loss_hat > delta
  false_alarm   fraction of truly safe moves with p_loss_hat > delta
  yield         consistent boards returned / K
  ms            sampling time per position

    python -m research.belief_eval --positions 400
    python -m research.belief_eval --positions 400 --weights submission/weights
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import chess

from . import ROOT
import features
from belief import DeterministicBelief, DiffusionBelief, HeuristicBelief
from .agents import build, sample_spec
from .selfplay import play_game

ALPHA = 0.5


def loses(board: chess.Board, move: chess.Move, side: chess.Color) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    king = after.king(side)
    return king is None or after.is_attacked_by(not side, king)


def true_class(board: chess.Board, square: int) -> int:
    piece = board.piece_at(square)
    return features.EMPTY if piece is None else features.ENEMY_TYPES.index(piece.piece_type) + 1


def score(belief, board, tracker, k, rng, delta):
    side = tracker.side
    began = time.monotonic()
    samples = belief.sample(tracker, k, rng)
    elapsed = time.monotonic() - began
    boards = samples or [tracker.visible_board()]
    hidden = [sq for sq, symbol in enumerate(tracker.current) if symbol == "?"]
    n = len(boards)
    result = {"yield": len(samples) / min(k, getattr(belief, "max_samples", k)), "ms": 1000 * elapsed}

    if hidden:
        nll = 0.0
        king_counts = defaultdict(int)
        for square in hidden:
            counts = [0] * features.NUM_CLASSES
            for sample in boards:
                cls = true_class(sample, square)
                counts[cls] += 1
                if cls == 6:
                    king_counts[square] += 1
            nll -= math.log((counts[true_class(board, square)] + ALPHA) / (n + ALPHA * features.NUM_CLASSES))
        result["square_nll"] = nll / len(hidden)
        king = board.king(not side)
        if king in hidden:
            result["king_nll"] = -math.log((king_counts[king] + ALPHA) / (n + ALPHA * len(hidden)))
            result["king_top1"] = float(bool(king_counts) and max(king_counts, key=king_counts.get) == king)

    brier, lossy, caught, safe, alarms = 0.0, 0, 0, 0, 0
    moves = tracker.legal_moves()
    for move in moves:
        predicted = sum(loses(sample, move, side) for sample in boards) / n
        actual = loses(board, move, side)
        brier += (predicted - actual) ** 2
        if actual:
            lossy += 1
            caught += predicted > delta
        else:
            safe += 1
            alarms += predicted > delta
    result["ploss_brier"] = brier / max(1, len(moves))
    if lossy:
        result["loss_recall"] = caught / lossy
    if safe:
        result["false_alarm"] = alarms / safe
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=400)
    parser.add_argument("--samples", type=int, default=32, help="K boards per position")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--turn-prob", type=float, default=0.25, help="chance a mover turn is scored")
    parser.add_argument("--weights", type=Path, default=None, help="add DiffusionBelief (fallback off)")
    parser.add_argument("--seed", type=int, default=9_000_000)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "belief_eval.json")
    args = parser.parse_args()

    beliefs = {"deterministic": DeterministicBelief(), "heuristic": HeuristicBelief()}
    if args.weights is not None:
        beliefs["diffusion"] = DiffusionBelief(args.weights, fallback=False, threads=2)
    totals = {name: defaultdict(list) for name in beliefs}
    plies: list[int] = []
    scored = 0
    game = 0
    pick = random.Random(args.seed)

    def on_turn(board, tracker):
        nonlocal scored
        if scored >= args.positions or pick.random() > args.turn_prob:
            return
        for name, belief in beliefs.items():
            for key, value in score(belief, board, tracker, args.samples,
                                    random.Random(pick.getrandbits(32)), args.delta).items():
                totals[name][key].append(value)
        plies.append(len(board.move_stack))
        scored += 1

    while scored < args.positions:
        rng = random.Random(args.seed + game)
        white, black = sample_spec(rng, held_out=True), sample_spec(rng, held_out=True)
        play_game(build(white), build(black), rng, max_plies=400, on_turn=on_turn)
        game += 1

    summary = {name: {key: sum(v) / len(v) for key, v in metrics.items()}
               for name, metrics in totals.items()}
    keys = ["square_nll", "king_nll", "king_top1", "ploss_brier", "loss_recall", "false_alarm", "yield", "ms"]
    print(f"positions={scored} games={game} mean_ply={sum(plies) / len(plies):.0f} "
          f"K={args.samples} delta={args.delta}")
    print(f"{'belief':14s}" + "".join(f"{key:>13s}" for key in keys))
    for name, metrics in summary.items():
        print(f"{name:14s}" + "".join(f"{metrics.get(key, float('nan')):13.4f}" for key in keys))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "positions": scored, "games": game, "summary": summary}, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

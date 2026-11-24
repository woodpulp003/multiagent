"""Generate (H_t, x_t) training pairs from in-process games (note, section 3.2).

The simulator keeps the true board; each agent receives only its observation.
At every turn we store the mover's encoded context (what the bot would see)
and the true labels of its hidden squares, plus who was playing.

    python -m research.selfplay --games 2000 --workers 6 --out data/round0
    python -m research.selfplay --games 200 --held-out --out data/heldout
    python -m research.selfplay --games 2000 --ours submission/weights --out data/round1
"""
from __future__ import annotations

import argparse
import multiprocessing
import random
from collections import Counter
from pathlib import Path

import chess
import numpy as np

from . import DATA
import features
from referee import FowBoard, fow_moves, observation, outcome, position_key
from tracker import GameTracker
from .agents import build, sample_spec


def play_game(white, black, rng: random.Random, max_plies: int = 1000, on_turn=None):
    """Play one game; return (examples, result). Each example is (context, tokens, labels, side, ply).

    `on_turn(board, tracker)` is called before every move with the true board
    and the mover's tracker (used by research.belief_eval).
    """
    board = FowBoard()
    agents = {chess.WHITE: white, chess.BLACK: black}
    recorders = {side: GameTracker(side) for side in agents}
    for side, agent in agents.items():
        agent.reset(side)
    repetitions = Counter({position_key(board): 1})
    examples = []
    while True:
        terminal = outcome(board, repetitions)
        if terminal:
            return examples, terminal[0]
        if len(board.move_stack) >= max_plies:
            return examples, "1/2-1/2"
        for side in agents:
            view = observation(board, side)
            recorders[side].observe(view)
            agents[side].observe(view)
        side = board.turn
        recorder = recorders[side]
        examples.append((features.encode_context(recorder),
                         features.encode_tokens(recorder.current, side),
                         features.encode_labels(board, recorder.current, side),
                         side, len(board.move_stack)))
        if on_turn is not None:
            on_turn(board, recorder)
        move = agents[side].act(rng)
        if move is None or move not in fow_moves(board):
            return examples, "0-1" if side else "1-0"
        recorder.record_move(move)
        board.push(move)
        repetitions[position_key(board)] += 1


def run_one(job):
    game_id, held_out, ours, max_plies = job
    rng = random.Random(game_id)
    specs = {chess.WHITE: sample_spec(rng, held_out, ours), chess.BLACK: sample_spec(rng, held_out, ours)}
    examples, result = play_game(build(specs[chess.WHITE]), build(specs[chess.BLACK]), rng, max_plies)
    names = {side: spec.name for side, spec in specs.items()}
    rows = [(context, tokens, labels, game_id, ply, names[side], names[not side])
            for context, tokens, labels, side, ply in examples]
    return rows, (names[chess.WHITE], names[chess.BLACK], result)


def save_shard(path: Path, rows):
    context, tokens, labels, game_id, ply, agent, opponent = zip(*rows)
    # Context planes lie in [0, 1]; uint8 halves memory so millions of rows fit in RAM.
    np.savez_compressed(path, context=np.round(np.stack(context) * 255).astype(np.uint8),
                        tokens=np.stack(tokens).astype(np.int8),
                        labels=np.stack(labels).astype(np.int8),
                        game_id=np.asarray(game_id, dtype=np.int64),
                        ply=np.asarray(ply, dtype=np.int16),
                        agent=np.asarray(agent), opponent=np.asarray(opponent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DATA / "round0")
    parser.add_argument("--seed", type=int, default=0, help="game ids start at seed * 1e6")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--held-out", action="store_true", help="use only held-out styles (evaluation data)")
    parser.add_argument("--ours", default=None, help="diffusion weights dir: add our bot to the pool")
    parser.add_argument("--games-per-shard", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed * 1_000_000 + g, args.held_out, args.ours, args.max_plies) for g in range(args.games)]
    rows, results, shard = [], Counter(), 0
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for done, (game_rows, (white, black, result)) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows += game_rows
            results[result] += 1
            if done % args.games_per_shard == 0 or done == len(jobs):
                save_shard(args.out / f"shard_{args.seed:04d}_{shard:04d}.npz", rows)
                print(f"games={done} examples_in_shard={len(rows)} results={dict(results)}", flush=True)
                rows, shard = [], shard + 1


if __name__ == "__main__":
    main()

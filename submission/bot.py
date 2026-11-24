"""Fog of War chess bot: belief sampling + risk-constrained move selection.

Protocol (see the starter README): read "fow" and "<w|b> <base> <increment>",
print "ready" within 15 s, then answer each numeric clock line with one
"move <uci>" line. stdout carries protocol lines only; diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

started = time.monotonic()

from belief import load_belief, log
from features import OBSERVATION_CHARACTERS
from planner import PlannerParams, choose_move, time_budget
from tracker import GameTracker

WEIGHTS = Path(__file__).resolve().parent / "weights"


def decide(tracker, belief, rng, remaining, increment, samples, delta):
    moves = tracker.legal_moves()
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]
    begin = time.monotonic()
    budget = time_budget(remaining, increment)
    boards = belief.sample(tracker, samples, rng, deadline=begin + 0.6 * budget)
    if not boards:
        boards = [tracker.visible_board()]
    move, stats = choose_move(moves, boards, tracker.current, tracker.side, rng,
                              deadline=begin + budget, params=PlannerParams(delta=delta))
    log(f"ply={tracker.ply} move={move.uci()} p_loss={stats.p_loss:.2f} q={stats.q:.0f} "
        f"samples={len(boards)} t={time.monotonic() - begin:.2f}s")
    return move


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--belief", choices=["auto", "heuristic", "deterministic", "diffusion"], default="auto")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if sys.stdin.readline().strip().lower() != "fow":
        return
    header = sys.stdin.readline().split()
    if len(header) != 3 or header[0].lower() not in ("w", "b"):
        raise ValueError("Expected color, base time, and increment")
    side = header[0].lower() == "w"
    increment = float(header[2])
    tracker = GameTracker(side)
    belief = load_belief(WEIGHTS, args.belief)
    if hasattr(belief, "warmup"):
        belief.warmup()
    log(f"init belief={belief.name} side={'w' if side else 'b'} t={time.monotonic() - started:.2f}s")
    print("ready", flush=True)

    for raw in sys.stdin:
        line = raw.strip()
        if len(line) == 64 and set(line) <= OBSERVATION_CHARACTERS:
            tracker.observe(line)
            continue
        try:
            remaining = float(line)
        except ValueError:
            continue
        try:
            move = decide(tracker, belief, rng, remaining, increment, args.samples, args.delta)
        except Exception as error:
            log(f"decide failed, falling back to a random legal move: {error!r}")
            moves = tracker.legal_moves()
            move = rng.choice(moves) if moves else None
        if move is None:
            return
        tracker.record_move(move)
        print("move " + move.uci(), flush=True)


if __name__ == "__main__":
    main()

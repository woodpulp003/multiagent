"""Risk-constrained move selection over sampled boards (note, section 4).

  p_loss(a) = mean_k L(s_k, a)     L = 1 if our king is capturable after a
  Q(a)      = mean_k Q(s_k, a)     Q from evaluation.Evaluator
  a*        = argmax Q(a) subject to p_loss(a) <= delta,
              else the move with the smallest p_loss, ties broken by Q.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import chess

from evaluation import EvalParams, Evaluator


@dataclass(frozen=True)
class PlannerParams:
    delta: float = 0.05            # tolerated estimated probability of immediate king loss
    temperature: float = 0.0       # 0 = argmax; >0 samples safe moves by softmax(Q / temperature)
    eval: EvalParams = field(default_factory=EvalParams)


@dataclass
class MoveStats:
    move: chess.Move
    losses: int = 0
    value: float = 0.0
    count: int = 0

    @property
    def p_loss(self) -> float:
        return self.losses / self.count if self.count else 1.0

    @property
    def q(self) -> float:
        return self.value / self.count if self.count else float("-inf")


def choose_move(moves: list[chess.Move], samples: list[chess.Board], view: str, side: chess.Color,
                rng: random.Random, deadline: float,
                params: PlannerParams = PlannerParams()) -> tuple[chess.Move, MoveStats]:
    evaluator = Evaluator(view, side, params.eval)
    stats = {move: MoveStats(move) for move in moves}
    for index, board in enumerate(samples):
        if index and time.monotonic() > deadline:
            break
        for move, lost, value in evaluator.evaluate_sample(board, moves):
            entry = stats[move]
            entry.losses += lost
            entry.value += value
            entry.count += 1
    entries = list(stats.values())
    safe = [e for e in entries if e.p_loss <= params.delta]
    if not safe:
        best = min(entries, key=lambda e: (e.p_loss, -e.q, rng.random()))
    elif params.temperature > 0:
        top = max(e.q for e in safe)
        weights = [math.exp((e.q - top) / params.temperature) for e in safe]
        best = rng.choices(safe, weights)[0]
    else:
        best = max(safe, key=lambda e: (e.q, rng.random()))
    return best.move, best


def time_budget(remaining: float, increment: float) -> float:
    """Seconds to spend on this move; conservative so 180+2 never flags."""
    return max(0.05, min(remaining / 30 + 0.75 * increment, 0.2 * remaining, 6.0))

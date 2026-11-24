"""Observation-only agents and the opponent pool used to generate training games.

Every agent sees exactly what a real bot would: its observations and its own
moves. Diversity comes from scripted bots, planner "personalities" (sampled
EvalParams/PlannerParams around named styles), and Stockfish playing on a
guessed board. Some styles are held out from training for evaluation.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

from . import ROOT
from belief import DiffusionBelief, HeuristicBelief
from evaluation import SIMPLE_VALUE, EvalParams
from planner import PlannerParams, choose_move
from tracker import GameTracker

STOCKFISH = Path(os.environ.get("FOW_STOCKFISH", ROOT / ".tools" / "stockfish" / "stockfish-macos-universal"))


class Agent:
    name = "agent"

    def reset(self, side: chess.Color):
        self.tracker = GameTracker(side)

    def observe(self, view: str):
        self.tracker.observe(view)

    def act(self, rng: random.Random) -> chess.Move | None:
        moves = self.tracker.legal_moves()
        move = self.pick(moves, rng) if moves else None
        if move is not None:
            self.tracker.record_move(move)
        return move

    def pick(self, moves, rng):
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def pick(self, moves, rng):
        return rng.choice(moves)


class GreedyAgent(Agent):
    """Captures the most valuable visible piece, otherwise random (like baseline.py)."""
    name = "greedy"

    def pick(self, moves, rng):
        view = self.tracker.current

        def score(move):
            target = view[move.to_square]
            value = SIMPLE_VALUE[chess.Piece.from_symbol(target).piece_type] if target not in "?-" else 0
            value += 0 if move.promotion is None else SIMPLE_VALUE[move.promotion] // 2
            return value + rng.random()
        return max(moves, key=score)


def make_belief(kind: str, weights: Path | None = None):
    if kind == "heuristic":
        return HeuristicBelief()
    if kind == "diffusion":
        return DiffusionBelief(weights, threads=1)
    if kind == "visible":
        return None
    raise ValueError(kind)


class PlannerAgent(Agent):
    """Our planner with configurable value weights, risk, temperature and belief."""

    def __init__(self, params: PlannerParams = PlannerParams(), belief: str = "heuristic",
                 samples: int = 8, random_opening: int = 0, weights: Path | None = None,
                 name: str = "planner", think_seconds: float = 1.0):
        self.params, self.samples, self.random_opening = params, samples, random_opening
        self.belief = make_belief(belief, weights)
        self.name, self.think_seconds = name, think_seconds

    def boards(self, rng):
        if self.belief is None:
            return [self.tracker.visible_board()]
        return self.belief.sample(self.tracker, self.samples, rng) or [self.tracker.visible_board()]

    def pick(self, moves, rng):
        if len(self.tracker.own_moves) < self.random_opening:
            return rng.choice(moves)
        return choose_move(moves, self.boards(rng), self.tracker.current, self.tracker.side, rng,
                           deadline=time.monotonic() + self.think_seconds, params=self.params)[0]


_ENGINES: dict[str, chess.engine.SimpleEngine] = {}


def engine(path: Path) -> chess.engine.SimpleEngine:
    """One persistent Stockfish per worker process."""
    key = str(path)
    if key not in _ENGINES:
        _ENGINES[key] = chess.engine.SimpleEngine.popen_uci(key, timeout=10.0)
    return _ENGINES[key]


def close_engines():
    while _ENGINES:
        _, sf = _ENGINES.popitem()
        try:
            sf.close()
        except Exception:
            pass


# python-chess drives each engine from a non-daemon thread, and the interpreter
# joins those threads *before* atexit handlers run, so an atexit quit() never
# fires and the process hangs on exit. This hook runs before that join.
threading._register_atexit(close_engines)


class StockfishAgent(PlannerAgent):
    """Stockfish on one guessed board; falls back to the planner when its move is unusable."""

    def __init__(self, skill: int = 10, movetime: float = 0.02, **planner_kwargs):
        super().__init__(**planner_kwargs)
        self.skill, self.movetime = skill, movetime
        self.name = f"stockfish{skill}"

    def pick(self, moves, rng):
        if len(self.tracker.own_moves) < self.random_opening:
            return rng.choice(moves)
        boards = self.boards(rng)
        board = chess.Board(boards[0].fen())
        board.castling_rights = self.tracker.rights  # opponent's rights are unknown
        if board.is_valid():
            try:
                sf = engine(STOCKFISH)
                sf.configure({"Skill Level": self.skill})
                move = sf.play(board, chess.engine.Limit(time=self.movetime)).move
                if move in moves:
                    return move
            except (chess.engine.EngineError, TimeoutError):
                close_engines()
        return choose_move(moves, boards, self.tracker.current, self.tracker.side, rng,
                           deadline=time.monotonic() + self.think_seconds, params=self.params)[0]


# --- Opponent pool -----------------------------------------------------------

STYLES: dict[str, dict] = {
    "balanced": {},
    "cautious": {"delta": 0.0, "king_exposure": 15.0},
    "reckless": {"delta": 0.4, "king_exposure": 0.0, "aggression": 0.3},
    "materialist": {"position": 0.3},
    "no_lookahead": {"capture_reply": 0.0},
    "positional": {"material": 0.8, "position": 2.0},
    "king_hunter": {"king_attack": 200.0, "aggression": 0.2, "delta": 0.15},
}
HELD_OUT_STYLES = ("positional", "king_hunter")


@dataclass
class AgentSpec:
    kind: str                                  # random | greedy | personality | stockfish | ours
    style: str = "balanced"
    params: PlannerParams = field(default_factory=PlannerParams)
    belief: str = "heuristic"
    samples: int = 8
    random_opening: int = 0
    skill: int = 10
    weights: str | None = None

    @property
    def name(self) -> str:
        if self.kind in ("random", "greedy"):
            return self.kind
        if self.kind == "stockfish":
            return f"stockfish{self.skill}"
        return f"{self.kind}:{self.style}:{self.belief}"


def sample_personality(rng: random.Random, style: str) -> PlannerParams:
    overrides = STYLES[style]
    defaults = EvalParams()

    def jitter(name, base):
        value = overrides.get(name, base)
        return value * rng.lognormvariate(0, 0.25) if value else value
    evaluation = EvalParams(**{name: jitter(name, getattr(defaults, name))
                               for name in EvalParams.__dataclass_fields__})
    delta = min(0.6, overrides.get("delta", 0.05) * rng.lognormvariate(0, 0.4))
    temperature = rng.choice((0.0, 0.0, 15.0, 50.0))
    return PlannerParams(delta=delta, temperature=temperature, eval=evaluation)


def sample_spec(rng: random.Random, held_out: bool = False, ours: str | None = None) -> AgentSpec:
    """Draw one opponent. `ours` is a diffusion weights dir to include our current bot."""
    styles = [s for s in STYLES if (s in HELD_OUT_STYLES) == held_out]
    roll = rng.random()
    if ours is not None and roll < 0.25:
        return AgentSpec("ours", params=PlannerParams(), belief="diffusion", samples=16, weights=ours)
    if roll < 0.35:
        return AgentSpec(rng.choice(("random", "greedy")))
    if roll < 0.55 and STOCKFISH.exists():
        return AgentSpec("stockfish", skill=rng.randint(0, 20), random_opening=rng.randint(0, 3),
                         samples=1)
    style = rng.choice(styles)
    return AgentSpec("personality", style=style, params=sample_personality(rng, style),
                     belief=rng.choice(("heuristic", "heuristic", "visible")),
                     samples=rng.choice((1, 4, 8, 16)), random_opening=rng.randint(0, 4))


def build(spec: AgentSpec) -> Agent:
    weights = Path(spec.weights) if spec.weights else None
    if spec.kind == "random":
        return RandomAgent()
    if spec.kind == "greedy":
        return GreedyAgent()
    common = dict(params=spec.params, belief=spec.belief, samples=spec.samples,
                  random_opening=spec.random_opening, weights=weights)
    if spec.kind == "stockfish":
        return StockfishAgent(skill=spec.skill, **common)
    return PlannerAgent(name=spec.name, **common)


def ours_spec(weights: str | None = None) -> AgentSpec:
    """Our deployed configuration (fixed value function, default risk)."""
    return AgentSpec("ours", style="deployed", belief="diffusion" if weights else "heuristic",
                     samples=32, weights=weights)

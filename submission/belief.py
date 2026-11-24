"""Belief samplers: produce complete boards consistent with the observation history.

Every sampler returns boards whose regenerated observation equals the real
one, so the planner can treat each sample as a possible true state.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

import features
from features import EMPTY, MASK, NUM_CLASSES, orient, piece_of
from referee import FowBoard, observation


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def pawn_constraints(tracker) -> tuple[set[int], set[int]]:
    """Hidden squares our pawns prove empty or occupied.

    Any enemy piece on a pawn's capture diagonal would be visible, so a hidden
    diagonal is empty. A hidden square in front of a pawn must be blocked,
    because an empty one would be visible.
    """
    view, side = tracker.current, tracker.side
    pawn, step = ("P", 8) if side else ("p", -8)
    start_rank = 1 if side else 6
    must_empty, must_occupied = set(), set()
    for square, symbol in enumerate(view):
        if symbol != pawn:
            continue
        forward = square + step
        if not 0 <= forward < 64:
            continue
        if view[forward] == "?":
            must_occupied.add(forward)
        elif view[forward] == "-" and chess.square_rank(square) == start_rank:
            if view[forward + step] == "?":
                must_occupied.add(forward + step)
        for df in (-1, 1):
            file = chess.square_file(square) + df
            if 0 <= file < 8 and view[chess.square(file, chess.square_rank(forward))] == "?":
                must_empty.add(chess.square(file, chess.square_rank(forward)))
    return must_empty, must_occupied - must_empty


def is_consistent(board: FowBoard, tracker) -> bool:
    """Exact check: the sample reproduces our observation and is a reachable turn."""
    enemy = not tracker.side
    kings = board.pieces(chess.KING, enemy)
    if len(kings) != 1:
        return False
    if board.is_attacked_by(tracker.side, next(iter(kings))):
        return False  # the opponent would already have lost
    return observation(board, tracker.side) == tracker.current


class HeuristicBelief:
    """Rules-based sampler: remembered pieces stay put, missing material is scattered.

    Serves as the non-learned comparison point and as the fallback whenever no
    trained diffusion weights are packaged.
    """
    name = "heuristic"

    def __init__(self, attempts_per_sample: int = 8):
        self.attempts_per_sample = attempts_per_sample

    def sample(self, tracker, k: int, rng: random.Random, deadline: float | None = None):
        base = tracker.visible_board()
        hidden = [sq for sq, symbol in enumerate(tracker.current) if symbol == "?"]
        must_empty, must_occupied = pawn_constraints(tracker)
        samples = []
        for _ in range(k * self.attempts_per_sample):
            if len(samples) >= k or (deadline is not None and time.monotonic() > deadline):
                break
            board = self.draw(tracker, base, hidden, must_empty, must_occupied, rng)
            if is_consistent(board, tracker):
                samples.append(board)
        return samples

    def draw(self, tracker, base, hidden, must_empty, must_occupied, rng):
        side, enemy, view, ply = tracker.side, not tracker.side, tracker.current, tracker.ply
        board = base.copy(stack=False)
        remaining = tracker.enemy_remaining()
        for symbol in view:
            if symbol not in "?-" and chess.Piece.from_symbol(symbol).color == enemy:
                piece_type = chess.Piece.from_symbol(symbol).piece_type
                remaining[piece_type] = max(0, remaining[piece_type] - 1)
        free = [sq for sq in hidden if sq not in must_empty]

        def relative_rank(square):
            return chess.square_rank(orient(square, side))

        def allowed(square, piece_type):
            return piece_type != chess.PAWN or 0 < relative_rank(square) < 7

        def place(square, piece_type):
            board.set_piece_at(square, chess.Piece(piece_type, enemy))
            free.remove(square)
            remaining[piece_type] -= 1

        def weighted_square(piece_type):
            options = [sq for sq in free if allowed(sq, piece_type)]
            if not options:
                return None
            weights = [1.0 + relative_rank(sq) for sq in options]
            if piece_type == chess.KING:
                weights = [w * (6.0 if relative_rank(sq) == 7 else 1.0) for w, sq in zip(weights, options)]
            return rng.choices(options, weights)[0]

        if remaining[chess.KING] > 0:
            remembered = [sq for sq in free if tracker.last_seen[sq] in ("k", "K")]
            square = remembered[0] if remembered and rng.random() < 0.75 else weighted_square(chess.KING)
            if square is not None:
                place(square, chess.KING)
        for square in list(free):
            symbol = tracker.last_seen[square]
            if symbol is None or symbol.lower() == "k":
                continue
            piece_type = chess.Piece.from_symbol(symbol).piece_type
            age = ply - tracker.last_seen_ply[square]
            if remaining[piece_type] > 0 and rng.random() < 0.85 ** age:
                place(square, piece_type)
        for square in must_occupied:
            if board.piece_at(square) is None and square in free:
                types = [t for t in remaining if t != chess.KING and remaining[t] > 0 and allowed(square, t)]
                place(square, rng.choice(types) if types else chess.PAWN)
        survival = max(0.3, 1.0 - ply / 150)
        for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            for _ in range(max(0, remaining[piece_type])):
                if rng.random() < survival:
                    square = weighted_square(piece_type)
                    if square is not None:
                        place(square, piece_type)
        return board


class DeterministicBelief:
    """Single most-likely board (the note's first comparison point).

    Remembered enemy pieces stay where last seen; never-seen squares keep their
    starting-position piece; the king defaults to its start square. Returns at
    most one board regardless of k.
    """
    name = "deterministic"
    max_samples = 1

    def sample(self, tracker, k: int, rng: random.Random, deadline: float | None = None):
        side, enemy, view = tracker.side, not tracker.side, tracker.current
        board = tracker.visible_board()
        must_empty, must_occupied = pawn_constraints(tracker)
        remaining = tracker.enemy_remaining()
        for symbol in view:
            if symbol not in "?-" and chess.Piece.from_symbol(symbol).color == enemy:
                piece_type = chess.Piece.from_symbol(symbol).piece_type
                remaining[piece_type] = max(0, remaining[piece_type] - 1)
        free = {sq for sq, symbol in enumerate(view) if symbol == "?" and sq not in must_empty}
        start = chess.Board()

        def place(square, piece_type):
            if square in free and remaining[piece_type] > 0:
                board.set_piece_at(square, chess.Piece(piece_type, enemy))
                free.discard(square)
                remaining[piece_type] -= 1

        for square in sorted(free, key=lambda sq: tracker.last_seen_ply[sq], reverse=True):
            symbol = tracker.last_seen[square]
            if symbol is not None:
                place(square, chess.Piece.from_symbol(symbol).piece_type)
        if remaining[chess.KING] > 0:
            king_start = chess.E8 if side else chess.E1
            if king_start in free:
                place(king_start, chess.KING)
            elif free:
                back = max(free, key=lambda sq: (chess.square_rank(orient(sq, side)),
                                                 -abs(chess.square_file(sq) - 4)))
                place(back, chess.KING)
        for square in sorted(free):
            piece = start.piece_at(square)
            if piece is not None and piece.color == enemy and tracker.last_visible_ply[square] < 0:
                place(square, piece.piece_type)
        for square in must_occupied:
            if square in free:
                rank = chess.square_rank(orient(square, side))
                board.set_piece_at(square, chess.Piece(chess.PAWN if 0 < rank < 7 else chess.KNIGHT, enemy))
                free.discard(square)
        return [board] if is_consistent(board, tracker) else []


class DiffusionBelief:
    """Masked discrete diffusion sampler backed by an exported ONNX denoiser.

    Model contract (see research/model.py and features.py):
      inputs  context float32[B, CONTEXT_PLANES, 8, 8], tokens int64[B, 64], t float32[B]
      output  logits  float32[B, 64, NUM_CLASSES]
    """
    name = "diffusion"

    def __init__(self, weights_dir: Path, steps: int | None = None, threads: int = 2,
                 fallback: bool = True):
        import onnxruntime as ort
        self.meta = json.loads((weights_dir / "meta.json").read_text())
        if self.meta.get("format_version") != features.FORMAT_VERSION:
            raise ValueError("Weights were trained with a different features.FORMAT_VERSION")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(weights_dir / "denoiser.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.steps = steps or int(self.meta.get("steps", 8))
        self.fallback = HeuristicBelief() if fallback else None

    def warmup(self):
        self.session.run(None, {
            "context": np.zeros((1, features.CONTEXT_PLANES, 8, 8), np.float32),
            "tokens": np.full((1, 64), MASK, np.int64),
            "t": np.ones(1, np.float32)})

    def allowed_classes(self, tracker) -> np.ndarray:
        allowed = np.ones((64, NUM_CLASSES), dtype=bool)
        must_empty, must_occupied = pawn_constraints(tracker)
        for square in must_empty:
            allowed[orient(square, tracker.side), 1:] = False
        for square in must_occupied:
            allowed[orient(square, tracker.side), EMPTY] = False
        for index in list(range(8)) + list(range(56, 64)):
            allowed[index, 1] = False  # pawns never stand on the first or last rank
        if any(symbol == ("k" if tracker.side else "K") for symbol in tracker.current):
            allowed[:, 6] = False
        return allowed

    def generate(self, tracker, k: int, rng: random.Random) -> np.ndarray:
        """Run the reverse chain; returns int64[k, 64] side-relative class tokens."""
        np_rng = np.random.default_rng(rng.getrandbits(64))
        context = np.repeat(features.encode_context(tracker)[None], k, axis=0)
        tokens = np.repeat(features.encode_tokens(tracker.current, tracker.side)[None], k, axis=0)
        allowed = self.allowed_classes(tracker)[None]
        for step in range(self.steps):
            t_now = 1.0 - step / self.steps
            t_next = 1.0 - (step + 1) / self.steps
            logits = self.session.run(None, {"context": context, "tokens": tokens,
                                             "t": np.full(k, t_now, np.float32)})[0]
            logits = np.where(allowed, logits, -np.inf)
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            draws = (probs.cumsum(axis=-1) > np_rng.random(probs.shape[:-1] + (1,))).argmax(axis=-1)
            masked = tokens == MASK
            reveal = masked if step == self.steps - 1 else masked & (
                np_rng.random(tokens.shape) < (t_now - t_next) / t_now)
            tokens = np.where(reveal, draws, tokens)
        return tokens

    def to_board(self, tracker, tokens: np.ndarray, rng: random.Random) -> FowBoard:
        board = tracker.visible_board()
        side = tracker.side
        kings = []
        for square, symbol in enumerate(tracker.current):
            if symbol != "?":
                continue
            piece = piece_of(int(tokens[orient(square, side)]), side)
            if piece is not None:
                board.set_piece_at(square, piece)
                if piece.piece_type == chess.KING:
                    kings.append(square)
        # TODO: smarter repair (material caps, choosing the king square by likelihood).
        for square in kings[1:] if len(kings) > 1 else []:
            board.remove_piece_at(square)
        return board

    def sample(self, tracker, k: int, rng: random.Random, deadline: float | None = None):
        samples = [board for board in (self.to_board(tracker, row, rng)
                                       for row in self.generate(tracker, k, rng))
                   if is_consistent(board, tracker)]
        if len(samples) < k and self.fallback is not None:
            samples += self.fallback.sample(tracker, k - len(samples), rng, deadline)
        return samples


def load_belief(weights_dir: Path, mode: str = "auto"):
    """'auto' uses diffusion only when trained weights are packaged."""
    onnx_path, meta_path = weights_dir / "denoiser.onnx", weights_dir / "meta.json"
    if mode == "heuristic":
        return HeuristicBelief()
    if mode == "deterministic":
        return DeterministicBelief()
    if mode == "diffusion":
        return DiffusionBelief(weights_dir)
    if onnx_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("trained", False):
            try:
                return DiffusionBelief(weights_dir)
            except Exception as error:  # never let a bad artifact cost the game
                log(f"diffusion unavailable, using heuristic: {error!r}")
    return HeuristicBelief()

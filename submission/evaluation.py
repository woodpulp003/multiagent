"""Value function Q(s, a) on one sampled full board (fixed across training rounds).

Q = PeSTO tapered material+position after our move
    - best capture reply available to the opponent (they can always see captures:
      a capture destination is revealed by their own move generation)
    - exposure of our king to hidden lines (observation-based, sample-independent)
    + small bonus for attacking the enemy king.

All weights live in EvalParams so research/ can derive opponent "personalities".
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

# PeSTO (Ronald Friederich) tables, a8..h1 order from White's point of view.
MG_VALUE = (82, 337, 365, 477, 1025, 0)
EG_VALUE = (94, 281, 297, 512, 936, 0)
PHASE_INC = (0, 1, 1, 2, 4, 0)

MG_TABLES = (
    (0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20,
     -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12,
     -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0),
    (-167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16,
     -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23),
    (-29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2,
     -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10,
     4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21),
    (32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
     -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33,
     -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26),
    (-28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57,
     -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5,
     -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50),
    (-65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22,
     -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27,
     1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14),
)
EG_TABLES = (
    (0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84,
     32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8,
     13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0),
    (-58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41,
     -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22,
     -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64),
    (-14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4,
     -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15,
     -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17),
    (13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3,
     4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16,
     -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20),
    (-9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9,
     3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5,
     -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41),
    (-74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9,
     -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43),
)

SIMPLE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000}
KNIGHT_DELTAS = (6, 10, 15, 17, -6, -10, -15, -17)
RAYS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
LOSS_VALUE = -100_000.0


@dataclass(frozen=True)
class EvalParams:
    material: float = 1.0          # scale of PeSTO piece values
    position: float = 1.0          # scale of PeSTO piece-square bonuses
    capture_reply: float = 1.0     # weight of the opponent's best capture reply (0 disables)
    king_exposure: float = 6.0     # penalty per hidden square with a line to our king
    king_attack: float = 40.0      # bonus when our move attacks the enemy king
    aggression: float = 0.0        # bonus per centipawn captured on top of material


def _entry(piece_type: int, color: chess.Color, square: int, params: EvalParams):
    index = square ^ 56 if color == chess.WHITE else square
    t = piece_type - 1
    mg = params.material * MG_VALUE[t] + params.position * MG_TABLES[t][index]
    eg = params.material * EG_VALUE[t] + params.position * EG_TABLES[t][index]
    return mg, eg, PHASE_INC[t]


class PestoState:
    """Side-relative tapered score of a board, updatable per move without rescanning."""
    __slots__ = ("mg", "eg", "phase")

    def __init__(self, board: chess.Board, side: chess.Color, params: EvalParams):
        self.mg = self.eg = 0.0
        self.phase = 0
        for square, piece in board.piece_map().items():
            mg, eg, phase = _entry(piece.piece_type, piece.color, square, params)
            sign = 1 if piece.color == side else -1
            self.mg += sign * mg
            self.eg += sign * eg
            self.phase += phase

    def after(self, board: chess.Board, move: chess.Move, side: chess.Color, params: EvalParams):
        """(mg, eg, phase) after `move` on `board` (side to move = `side`)."""
        mg, eg, phase = self.mg, self.eg, self.phase
        mover = board.piece_at(move.from_square)
        d_mg, d_eg, _ = _entry(mover.piece_type, side, move.from_square, params)
        mg, eg = mg - d_mg, eg - d_eg
        placed = move.promotion or mover.piece_type
        d_mg, d_eg, d_phase = _entry(placed, side, move.to_square, params)
        mg, eg, phase = mg + d_mg, eg + d_eg, phase + d_phase - PHASE_INC[mover.piece_type - 1]
        victim_square = move.to_square
        if board.is_en_passant(move):
            victim_square = move.to_square + (-8 if side else 8)
        victim = board.piece_at(victim_square)
        if victim is not None and victim.color != side:
            d_mg, d_eg, d_phase = _entry(victim.piece_type, victim.color, victim_square, params)
            mg, eg, phase = mg + d_mg, eg + d_eg, phase - d_phase
        if board.is_castling(move):
            rank = 0 if side else 7
            kingside = move.to_square > move.from_square
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            a_mg, a_eg, _ = _entry(chess.ROOK, side, rook_from, params)
            b_mg, b_eg, _ = _entry(chess.ROOK, side, rook_to, params)
            mg, eg = mg - a_mg + b_mg, eg - a_eg + b_eg
        return mg, eg, phase


def tapered(mg: float, eg: float, phase: int) -> float:
    phase = min(phase, 24)
    return (mg * phase + eg * (24 - phase)) / 24


def best_capture_gain(board: chess.Board, enemy: chess.Color) -> float:
    """Largest naive gain from one enemy capture (board has enemy to move)."""
    best = 0.0
    for move in board.generate_pseudo_legal_captures():
        victim = board.piece_at(move.to_square)
        victim_value = SIMPLE_VALUE[victim.piece_type] if victim else SIMPLE_VALUE[chess.PAWN]
        attacker_value = SIMPLE_VALUE[board.piece_at(move.from_square).piece_type]
        defended = board.is_attacked_by(not enemy, move.to_square)
        best = max(best, victim_value - (attacker_value if defended else 0))
    return best


def king_exposure(view: str, move: chess.Move, side: chess.Color) -> int:
    """Hidden squares on open lines or knight jumps to our king after `move`."""
    own = "PNBRQK" if side else "pnbrqk"
    squares = list(view)
    squares[move.to_square] = squares[move.from_square]
    squares[move.from_square] = "-"
    king = next((sq for sq, s in enumerate(squares) if s == own[5]), None)
    if king is None:
        return 0
    file, rank = chess.square_file(king), chess.square_rank(king)
    exposed = 0
    for df, dr in RAYS:
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            symbol = squares[chess.square(f, r)]
            if symbol == "?":
                exposed += 1
                break
            if symbol != "-":
                break
            f, r = f + df, r + dr
    for delta in KNIGHT_DELTAS:
        target = king + delta
        if 0 <= target < 64 and abs(chess.square_file(target) - file) <= 2 and squares[target] == "?":
            exposed += 1
    return exposed


class Evaluator:
    """Evaluates (sample, move) pairs; one instance per decision."""

    def __init__(self, view: str, side: chess.Color, params: EvalParams = EvalParams()):
        self.view, self.side, self.params = view, side, params
        self._exposure: dict[chess.Move, int] = {}

    def exposure(self, move: chess.Move) -> int:
        if move not in self._exposure:
            self._exposure[move] = king_exposure(self.view, move, self.side)
        return self._exposure[move]

    def evaluate_sample(self, board: chess.Board, moves: list[chess.Move]):
        """Yield (move, king capturable after move, value) for every move on one sample."""
        side, enemy, params = self.side, not self.side, self.params
        state = PestoState(board, side, params)
        for move in moves:
            mg, eg, phase = state.after(board, move, side, params)
            victim = board.piece_at(move.to_square)
            after = board.copy(stack=False)
            after.push(move)
            king = after.king(side)
            if king is None or after.is_attacked_by(enemy, king):
                yield move, True, LOSS_VALUE
                continue
            value = tapered(mg, eg, phase)
            if params.capture_reply:
                value -= params.capture_reply * best_capture_gain(after, enemy)
            value -= params.king_exposure * self.exposure(move)
            enemy_king = after.king(enemy)
            if params.king_attack and enemy_king is not None and after.is_attacked_by(side, enemy_king):
                value += params.king_attack
            if params.aggression and victim is not None and victim.color == enemy:
                value += params.aggression * SIMPLE_VALUE[victim.piece_type]
            yield move, False, value

"""Everything the bot is allowed to know: its observations and its own moves."""
from __future__ import annotations

import chess

from features import OBSERVATION_CHARACTERS
from referee import FowBoard
from uniform_random import candidates, infer_en_passant

INITIAL_COUNTS = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
                  chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}


class GameTracker:
    """Observation history, own castling rights, enemy memory and capture counts.

    One observation arrives before every ply, so ``ply`` (the index of the
    latest observation) equals the number of plies already played.
    """

    def __init__(self, side: chess.Color):
        self.side = side
        rank = 0 if side else 7
        self.king_start = chess.square(4, rank)
        self.rook_starts = (chess.square(0, rank), chess.square(7, rank))
        self.rights = sum(chess.BB_SQUARES[square] for square in self.rook_starts)
        self.observations: list[str] = []
        self.own_moves: list[tuple[int, chess.Move]] = []
        self.last_seen: list[str | None] = [None] * 64
        self.last_seen_ply = [-1] * 64
        self.last_visible_ply = [-1] * 64
        self.enemy_captured = {piece_type: 0 for piece_type in INITIAL_COUNTS}

    @property
    def ply(self) -> int:
        return len(self.observations) - 1

    @property
    def current(self) -> str | None:
        return self.observations[-1] if self.observations else None

    @property
    def previous(self) -> str | None:
        return self.observations[-2] if len(self.observations) > 1 else None

    def observe(self, view: str):
        if len(view) != 64 or not set(view) <= OBSERVATION_CHARACTERS:
            raise ValueError("Expected a 64-square FOW observation")
        self.observations.append(view)
        ply = self.ply
        king, rook = ("K", "R") if self.side else ("k", "r")
        if view[self.king_start] != king:
            self.rights = 0
        for square in self.rook_starts:
            if view[square] != rook:
                self.rights &= ~chess.BB_SQUARES[square]
        enemy_king = "k" if self.side else "K"
        for square, symbol in enumerate(view):
            if symbol == "?":
                continue
            self.last_visible_ply[square] = ply
            if symbol == "-" or chess.Piece.from_symbol(symbol).color == self.side:
                self.last_seen[square] = None
                continue
            if symbol == enemy_king:
                for other in range(64):
                    if self.last_seen[other] == enemy_king:
                        self.last_seen[other] = None
            self.last_seen[square] = symbol
            self.last_seen_ply[square] = ply

    def ep_square(self) -> int | None:
        try:
            return infer_en_passant(self.previous, self.current, self.side)
        except ValueError:
            return None

    def legal_moves(self) -> list[chess.Move]:
        if self.current is None:
            raise ValueError("No observation received")
        return candidates(self.current, self.side, self.rights, self.previous)[1]

    def visible_board(self) -> FowBoard:
        """Board with every visible piece, our rights and EP; hidden squares empty."""
        board = FowBoard(None)
        board.turn = self.side
        for square, symbol in enumerate(self.current):
            if symbol not in "?-":
                board.set_piece_at(square, chess.Piece.from_symbol(symbol))
        board.castling_rights = self.rights
        board.ep_square = self.ep_square()
        return board

    def record_move(self, move: chess.Move):
        if self.current is None:
            raise ValueError("No observation received")
        view = self.current
        target = view[move.to_square]
        if target not in "?-":
            self.enemy_captured[chess.Piece.from_symbol(target).piece_type] += 1
        elif view[move.from_square].lower() == "p" and move.to_square == self.ep_square():
            self.enemy_captured[chess.PAWN] += 1
        if view[move.from_square].lower() == "k":
            self.rights = 0
        self.rights &= ~chess.BB_SQUARES[move.from_square]
        self.own_moves.append((self.ply, move))

    def enemy_remaining(self) -> dict[int, int]:
        """Upper bound on enemy material (ignores promotions)."""
        return {t: INITIAL_COUNTS[t] - self.enemy_captured[t] for t in INITIAL_COUNTS}

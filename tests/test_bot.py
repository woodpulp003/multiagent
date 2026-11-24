import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pytest

import features
from belief import DeterministicBelief, HeuristicBelief, is_consistent
from referee import FowBoard, fow_moves, observation, outcome, position_key, run_match
from tracker import GameTracker

ROOT = Path(__file__).resolve().parent.parent
RUN = ["bash", str(ROOT / "submission" / "run.sh")]


def random_games(games, seed, max_plies=200):
    """Yield (true board, tracker of the side to move) at every turn of random games."""
    rng = random.Random(seed)
    for _ in range(games):
        board = FowBoard()
        trackers = {side: GameTracker(side) for side in (True, False)}
        repetitions = Counter({position_key(board): 1})
        for _ in range(max_plies):
            if outcome(board, repetitions):
                break
            for side, tracker in trackers.items():
                tracker.observe(observation(board, side))
            yield board, trackers[board.turn]
            move = rng.choice(fow_moves(board))
            trackers[board.turn].record_move(move)
            board.push(move)
            repetitions[position_key(board)] += 1


def test_tracker_legal_moves_match_referee():
    for board, tracker in random_games(20, seed=1):
        assert {m.uci() for m in tracker.legal_moves()} == {m.uci() for m in fow_moves(board)}


def test_true_board_passes_consistency_check():
    for board, tracker in random_games(20, seed=2):
        truth = board.copy(stack=False)
        truth.castling_rights = tracker.rights
        truth.ep_square = tracker.ep_square()
        assert is_consistent(truth, tracker), board.fen()


def test_heuristic_samples_reproduce_the_observation():
    rng = random.Random(3)
    belief = HeuristicBelief()
    produced = 0
    for board, tracker in random_games(5, seed=3):
        for sample in belief.sample(tracker, 4, rng):
            assert observation(sample, tracker.side) == tracker.current
            assert len(sample.pieces(chess.KING, not tracker.side)) == 1
            produced += 1
    assert produced > 100


def test_deterministic_belief_is_consistent_and_usually_succeeds():
    belief = DeterministicBelief()
    rng = random.Random(7)
    positions = produced = 0
    for board, tracker in random_games(5, seed=7):
        boards = belief.sample(tracker, 8, rng)
        assert len(boards) <= 1
        for sample in boards:
            assert observation(sample, tracker.side) == tracker.current
        positions += 1
        produced += len(boards)
    assert produced > 0.8 * positions


def test_encodings_have_contract_shapes_and_labels():
    for board, tracker in random_games(3, seed=4):
        context = features.encode_context(tracker)
        tokens = features.encode_tokens(tracker.current, tracker.side)
        labels = features.encode_labels(board, tracker.current, tracker.side)
        assert context.shape == (features.CONTEXT_PLANES, 8, 8)
        assert ((tokens == features.MASK) == (labels != features.IGNORE_LABEL)).all()
        assert labels[labels != features.IGNORE_LABEL].max(initial=0) < features.NUM_CLASSES


def test_protocol_ready_then_one_move():
    transcript = "fow\nw 180 2\n" + observation(FowBoard(), True) + "\n180\n"
    result = subprocess.run(RUN + ["--seed", "1"], input=transcript, text=True,
                            capture_output=True, check=True, timeout=15)
    lines = result.stdout.splitlines()
    assert lines[0] == "ready" and len(lines) == 2
    assert lines[1].removeprefix("move ") in {m.uci() for m in fow_moves(FowBoard())}


@pytest.mark.parametrize("our_color", ["white", "black"])
def test_full_game_against_baseline_without_failures(our_color):
    baseline = [sys.executable, "-u", str(ROOT / "opponents" / "baseline.py"), "--seed", "5"]
    white, black = (RUN, baseline) if our_color == "white" else (baseline, RUN)
    replay = run_match(white, black, base=60, increment=1, max_plies=400)
    assert "failedSide" not in replay, replay.get("studentReports")


def test_untrained_diffusion_weights_load_and_sample(tmp_path):
    pytest.importorskip("torch")
    subprocess.run([sys.executable, "-m", "research.export", "--untrained",
                    "--out", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    from belief import DiffusionBelief
    belief = DiffusionBelief(tmp_path, steps=4)
    belief.warmup()
    rng = random.Random(6)
    for _, tracker in random_games(1, seed=6, max_plies=30):
        tokens = belief.generate(tracker, 8, rng)
        assert tokens.shape == (8, 64) and (tokens != features.MASK).all()
        for sample in belief.sample(tracker, 8, rng):
            assert observation(sample, tracker.side) == tracker.current

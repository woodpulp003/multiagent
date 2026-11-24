"""Play our bot against reference opponents through the real referee, both colors.

    python -m research.eval --games 10 --belief heuristic --opponent baseline

Runs the bot via submission/run.sh exactly as the arena does, but with the
local venv's Python. Reports W/D/L from our point of view and any protocol failures.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections import Counter

from . import OPPONENTS, SUBMISSION
from referee import run_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10, help="games per color")
    parser.add_argument("--belief", default="auto", choices=["auto", "heuristic", "deterministic", "diffusion"])
    parser.add_argument("--opponent", default="baseline", choices=["baseline", "uniform_random"])
    parser.add_argument("--bot-args", default="", help="extra arguments for bot.py")
    parser.add_argument("--base-time", type=float, default=180)
    parser.add_argument("--increment", type=float, default=2)
    args = parser.parse_args()
    os.environ["FOW_PYTHON"] = sys.executable
    tally, reasons = Counter(), Counter()
    for game in range(args.games):
        for our_color in ("white", "black"):
            ours = ["bash", str(SUBMISSION / "run.sh"), "--belief", args.belief,
                    "--seed", str(game), *shlex.split(args.bot_args)]
            theirs = [sys.executable, "-u", str(OPPONENTS / f"{args.opponent}.py"), "--seed", str(10_000 + game)]
            white, black = (ours, theirs) if our_color == "white" else (theirs, ours)
            replay = run_match(white, black, args.base_time, args.increment, max_plies=1000)
            win = "1-0" if our_color == "white" else "0-1"
            outcome = "draw" if replay["result"] == "1/2-1/2" else "win" if replay["result"] == win else "loss"
            if replay.get("failedSide") == our_color:
                outcome = "FAILURE"
            tally[outcome] += 1
            reasons[(outcome, replay["reason"])] += 1
            print(f"game={game} color={our_color} {outcome}: {replay['reason']} "
                  f"({len(replay['frames']) - 1} plies)", flush=True)
    print(dict(tally))
    for (outcome, reason), count in sorted(reasons.items()):
        print(f"  {outcome:7s} {reason}: {count}")


if __name__ == "__main__":
    main()

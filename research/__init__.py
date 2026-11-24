"""Offline tooling (never uploaded). Run modules as `python -m research.<name>`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
OPPONENTS = ROOT / "opponents"
WEIGHTS = SUBMISSION / "weights"
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"

# The bot's modules (features, tracker, referee, ...) are the single source of
# truth for rules and encodings, so training can never drift from deployment.
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

"""Approximate the arena's validation on the packaged zip.

1. Build dist/fow-diffusion.zip with scripts/package.sh.
2. Extract it into a fresh temporary directory (nothing else from the repo).
3. Run build.sh if present (5-minute limit), then make the bundle read-only.
4. Play one full 180+2 game against baseline.py in each color. The bot runs
   with an empty environment: no PYTHONPATH, no FOW_PYTHON, and only the
   chosen interpreter's directory on PATH as `python3`.

Losing or drawing passes; crashes, protocol errors, illegal moves, or slow
initialization fail.

    .venv/bin/python scripts/validate.py --runtime-python .runtime/bin/python3
"""
from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "opponents"))
from referee import run_match  # noqa: E402


def make_read_only(directory: Path):
    for path in [directory, *directory.rglob("*")]:
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def make_writable(directory: Path):
    for path in [directory, *directory.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable),
                        help="interpreter exposed to the bot as python3")
    parser.add_argument("--base-time", type=float, default=180)
    parser.add_argument("--increment", type=float, default=2)
    args = parser.parse_args()
    subprocess.run(["bash", str(ROOT / "scripts" / "package.sh")], check=True)
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = tmp / "bot"
        with zipfile.ZipFile(ROOT / "dist" / "fow-diffusion.zip") as archive:
            archive.extractall(bundle)
        if not (bundle / "run.sh").is_file():
            sys.exit("FAIL: run.sh is not at the archive root")
        shim = tmp / "bin"
        shim.mkdir()
        (shim / "python3").write_text(f'#!/bin/sh\nexec "{args.runtime_python.absolute()}" "$@"\n')
        (shim / "python3").chmod(0o755)
        clean_env = ["env", "-i", f"HOME={tmp}", f"PATH={shim}:/usr/bin:/bin"]
        if (bundle / "build.sh").is_file():
            subprocess.run([*clean_env, "bash", "build.sh"], cwd=bundle, check=True, timeout=300)
        make_read_only(bundle)
        baseline = [sys.executable, "-u", str(ROOT / "opponents" / "baseline.py"), "--seed", "1"]
        ours = [*clean_env, "bash", str(bundle / "run.sh")]
        os.chdir(tmp)
        try:
            for color, (white, black) in (("white", (ours, baseline)), ("black", (baseline, ours))):
                replay = run_match(white, black, args.base_time, args.increment, max_plies=1000)
                report = replay.get("studentReports", [{}])[0]
                status = "FAIL" if replay.get("failedSide") == color else "ok"
                failed |= status == "FAIL"
                warnings = [w["code"] for w in replay.get("warnings", []) if w["side"] == color]
                print(f"{status}: as {color}: {replay['result']} {replay['reason']} "
                      f"({len(replay['frames']) - 1} plies) {report.get('code', '')} {warnings or ''}")
                if status == "FAIL":
                    print(replay.get("privateDiagnostics", {}).get(color, "")[-2000:])
        finally:
            os.chdir(ROOT)
            make_writable(bundle)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

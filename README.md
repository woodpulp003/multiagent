# Fog of War chess: conditional diffusion belief bot

MIT 6.7980 final project. The bot samples plausible hidden boards from its
observation history with a conditional discrete diffusion model, then picks
the best move whose estimated king-loss probability stays below a threshold
(see `../fow_diffusion_note.pdf`).

## Layout

```text
submission/           # the uploaded bot: its CONTENTS form the zip root
  run.sh              # arena entry point (required)
  bot.py              # stdin/stdout protocol loop
  tracker.py          # observation history, own castling rights, enemy memory
  features.py         # tensor encoding shared by bot and training (the model contract)
  belief.py           # HeuristicBelief (baseline/fallback), DiffusionBelief (ONNX)
  planner.py          # p_loss / Q evaluation and the risk-constrained argmax
  referee.py          # verbatim copy of the kit's rules engine
  uniform_random.py   # verbatim copy of the kit's starter (exact move generation, EP inference)
  weights/            # denoiser.onnx + meta.json, written by research/export.py
research/             # offline only, never uploaded
  agents.py           # observation-only bots for self-play
  selfplay.py         # generate (history, hidden-state) training data
  model.py            # PyTorch denoiser (SCAFFOLD architecture)
  train.py            # masked-diffusion cross-entropy training
  export.py           # checkpoint -> ONNX + meta.json, numerically checked
  eval.py             # matches vs reference bots through the real referee
opponents/            # kit's baseline.py and uniform_random.py, for testing
scripts/package.sh    # build dist/fow-diffusion.zip and enforce upload limits
scripts/validate.py   # mimic arena validation on the zip (both colors vs baseline)
tests/                # pytest suite, including the kit's own tests
```

## Setup

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Workflow

```sh
.venv/bin/python -m research.selfplay --games 200 --seed 0     # data/selfplay/*.npz
.venv/bin/python -m research.train --epochs 5                  # checkpoints/denoiser.pt
.venv/bin/python -m research.export                            # submission/weights/
.venv/bin/python -m research.eval --games 10 --belief diffusion
.venv/bin/python scripts/validate.py                           # then upload dist/fow-diffusion.zip
```

Without trained weights, `--belief auto` (the default in `run.sh`) uses the
heuristic sampler, so the submission is always valid.

## Submission rules this repo enforces

- `run.sh` sits at the zip root; no `build.sh` is needed because weights ship as ONNX.
- The bot imports only the standard library, `chess`, `numpy` and `onnxruntime`,
  all preinstalled. Never import `torch` in `submission/`: it is slow to load and
  the 15-second initialization limit includes model loading.
- stdout carries only `ready` and `move <uci>` lines; all logging goes to stderr
  (the arena keeps the last 16 KiB; over 1 MiB total or 4 KiB per line warns).
- Any exception while choosing falls back to a random legal move, because
  crashes fail validation and forfeit rated games.
- Time per move is capped by `planner.time_budget` (180 s + 2 s/move clock, the
  process is frozen during the opponent's turn).
- ONNX Runtime uses 2 intra-op threads on the arena's 2 physical cores.
- Limits: 128 MiB compressed, 512 MiB expanded, 10,000 entries, no symlinks,
  4 GiB memory, no network.

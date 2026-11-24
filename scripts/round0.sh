#!/bin/bash
# Round-0 data generation: held-out evaluation set first, then the training set.
# Resumable: each chunk writes DONE when complete, and finished chunks are skipped.
#
#   nohup caffeinate -is bash scripts/round0.sh >/dev/null 2>&1 &
#   tail -f data/round0/progress.log
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-data/round0}"
WORKERS="${WORKERS:-8}"
GAMES_PER_CHUNK="${GAMES_PER_CHUNK:-2000}"
TRAIN_CHUNKS="${TRAIN_CHUNKS:-30}"    # seeds 0..29   -> 60k games, ~2.5M positions
HELDOUT_CHUNKS="${HELDOUT_CHUNKS:-3}" # seeds 500..502 -> 6k games, held-out styles only
LOG="$OUT/progress.log"
mkdir -p "$OUT/train" "$OUT/heldout"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

run_chunk() {
    local split=$1 seed=$2; shift 2
    local dir
    dir="$OUT/$split/seed_$(printf %04d "$seed")"
    if [ -f "$dir/DONE" ]; then return 0; fi
    rm -rf "$dir"
    log "start $split seed=$seed"
    if "$PY" -m research.selfplay --games "$GAMES_PER_CHUNK" --workers "$WORKERS" \
            --games-per-shard 500 --seed "$seed" --out "$dir" "$@" >> "$LOG" 2>&1; then
        touch "$dir/DONE"
        log "done  $split seed=$seed"
    else
        log "FAILED $split seed=$seed (rerun the script to retry)"
    fi
}

log "round0 begin pid=$$ workers=$WORKERS games/chunk=$GAMES_PER_CHUNK"
for ((i = 0; i < HELDOUT_CHUNKS; i++)); do run_chunk heldout $((500 + i)) --held-out; done
for ((i = 0; i < TRAIN_CHUNKS; i++)); do run_chunk train "$i"; done
done_chunks=$(find "$OUT" -name DONE | wc -l | tr -d ' ')
log "round0 end: $done_chunks/$((TRAIN_CHUNKS + HELDOUT_CHUNKS)) chunks complete"

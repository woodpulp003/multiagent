#!/bin/bash
# Zip the contents of submission/ into dist/fow-diffusion.zip and enforce the
# upload rules: run.sh at the root, no symlinks, <=128 MiB compressed,
# <=512 MiB expanded, <=10,000 entries.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/submission"
out="$root/dist/fow-diffusion.zip"

fail() { echo "package: $*" >&2; exit 1; }

[ -f "$src/run.sh" ] || fail "submission/run.sh is missing"
if [ -n "$(find "$src" -type l)" ]; then
    fail "symlinks are not accepted: $(find "$src" -type l | tr '\n' ' ')"
fi

mkdir -p "$root/dist"
rm -f "$out"
(cd "$src" && zip -q -r -X "$out" . \
    -x '*__pycache__*' '*.pyc' '*.DS_Store' 'weights/.gitkeep' '.pytest_cache/*')

compressed=$(wc -c < "$out" | tr -d ' ')
read -r expanded entries < <(unzip -l "$out" | tail -n 1 | awk '{print $1, $2}')
[ "$compressed" -le $((128 * 1024 * 1024)) ] || fail "zip is $compressed bytes (> 128 MiB)"
[ "$expanded" -le $((512 * 1024 * 1024)) ] || fail "expanded size $expanded bytes (> 512 MiB)"
[ "$entries" -le 10000 ] || fail "$entries entries (> 10,000)"
unzip -Z1 "$out" | sed 's/^/  /'
echo "package: $out  compressed=${compressed}B expanded=${expanded}B entries=$entries"

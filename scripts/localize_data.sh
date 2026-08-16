#!/bin/bash
# Copy a staged dataset from slow/network storage (EFS/NFS) to fast local disk,
# then point the pipeline at the copy.
#
# Why: with cache_volumes off, training mmaps the .npy volumes and reads random
# patch pages. On EFS every page fault is a network round trip, which is
# latency-bound and slow; one sequential bulk copy is fast, and local mmap reads
# afterwards are nearly free. Run this ONCE per VM boot, then export DATA_DIR.
#
# Usage
# -----
#   bash scripts/localize_data.sh                          # copies default DATA_DIR
#   DATA_DIR=/efs/data_x LOCAL_ROOT=/nvme bash scripts/localize_data.sh
#   ...then:  export DATA_DIR=<printed path>  (or eval the printed export line)
#
# Env knobs (defaults in parens)
#   DATA_DIR    (data_phalloidin_mhc_051826_staged)  source staged root (bf/ gfp/ stats/)
#   LOCAL_ROOT  (/tmp/myo_local)                     local destination root
#
# Re-running is cheap: existing same-size files are skipped.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data_phalloidin_mhc_051826_staged}"
LOCAL_ROOT="${LOCAL_ROOT:-/tmp/myo_local}"

[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR '$DATA_DIR' not found" >&2; exit 1; }

DEST="$LOCAL_ROOT/$(basename "$DATA_DIR")"
mkdir -p "$DEST"

echo "localizing $DATA_DIR -> $DEST"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "$DATA_DIR/" "$DEST/"
else
  # cp fallback: skip files that already exist with the same size
  ( cd "$DATA_DIR" && find . -type f ) | while read -r f; do
    src="$DATA_DIR/$f"; dst="$DEST/$f"
    if [ -f "$dst" ] && [ "$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src")" \
        = "$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst")" ]; then
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  done
fi

du -sh "$DEST" | awk '{print "local copy: " $1}'
echo ""
echo "✅ done. Point the pipeline at the local copy:"
echo "   export DATA_DIR=\"$DEST\""
echo "   bash scripts/force_all.sh"

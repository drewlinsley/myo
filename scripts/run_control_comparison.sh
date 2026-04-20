#!/bin/bash
# GFP control vs BF->GFP + classifier, binary LOO for Exercise and Perturbation.
#
# Arms:
#   control : GFP fed through ImageNet-only encoder (no BF->GFP pretraining)
#   bfgfp   : BF fed through BF->GFP-pretrained encoder
#
# Both arms train a binary head (Control vs Perturbed) via LOO, with a
# permutation test (default 10k shuffles) on the fixed prediction vector.
#
# Usage:
#   bash scripts/run_control_comparison.sh
#   BF_CKPT=ckpts/unet_3d_imagenet_pearson_frac100/best.pth \
#       bash scripts/run_control_comparison.sh
#   FORCE=1 bash scripts/run_control_comparison.sh   # re-run even if JSON exists
#
# Idempotent: if the output JSON for an arm+task already exists, that run is
# skipped. Set FORCE=1 to retrain.

set -e

METADATA=data_mapping_drew.csv
LOO_CFG=configs/gfp_classifier_3d.yaml
BFGFP_CFG=configs/unet_3d_imagenet_pearson.yaml
LOO_OUT=results/loo
OUT_DIR=results/classifier
EXTRA_ARGS="$@"

mkdir -p "$LOO_OUT" "$OUT_DIR"

# Pick BF->GFP checkpoint: env override, else highest-fraction Step 2 output,
# else train frac=1.0 from scratch now.
if [ -z "$BF_CKPT" ]; then
    for cand in ckpts/*frac100*/best.pth \
                ckpts/*frac050*/best.pth \
                ckpts/*frac025*/best.pth \
                ckpts/*frac010*/best.pth \
                ckpts/*frac005*/best.pth; do
        if compgen -G "$cand" > /dev/null; then
            BF_CKPT=$(ls $cand | head -1)
            break
        fi
    done
fi

if [ -z "$BF_CKPT" ] || [ ! -f "$BF_CKPT" ]; then
    echo ""
    echo "No BF->GFP checkpoint found — training frac=1.0 now."
    echo "(Set BF_CKPT=path/to/best.pth to skip this.)"
    python train_fraction.py -c "$BFGFP_CFG" --fraction 1.0
    for cand in ckpts/*frac100*/best.pth; do
        if compgen -G "$cand" > /dev/null; then
            BF_CKPT=$(ls $cand | head -1)
            break
        fi
    done
    if [ -z "$BF_CKPT" ] || [ ! -f "$BF_CKPT" ]; then
        echo "ERROR: BF->GFP training finished but no best.pth found." >&2
        exit 1
    fi
fi
echo "Using BF->GFP checkpoint: $BF_CKPT"

# ──────────────────────────────────────────────────────────────
# Arm 1: GFP control (no BF->GFP encoder pretraining)
# ──────────────────────────────────────────────────────────────
for task in exercise perturbation; do
    out="$LOO_OUT/control_${task}_gfp.json"
    if [ -f "$out" ] && [ -z "$FORCE" ]; then
        echo "skip: $out exists (set FORCE=1 to retrain)"
        continue
    fi
    echo ""
    echo "=== control LOO ($task, GFP input, ImageNet encoder) ==="
    python train_loo_classifier.py \
        -c "$LOO_CFG" \
        --metadata "$METADATA" \
        --task "$task" --input gfp \
        --binarize \
        --output "$out" \
        $EXTRA_ARGS
done

# ──────────────────────────────────────────────────────────────
# Arm 2: BF->GFP encoder + classifier
# ──────────────────────────────────────────────────────────────
for task in exercise perturbation; do
    out="$LOO_OUT/bfgfp_${task}_bf.json"
    if [ -f "$out" ] && [ -z "$FORCE" ]; then
        echo "skip: $out exists (set FORCE=1 to retrain)"
        continue
    fi
    echo ""
    echo "=== bfgfp LOO ($task, BF input, init=$BF_CKPT) ==="
    python train_loo_classifier.py \
        -c "$LOO_CFG" \
        --metadata "$METADATA" \
        --task "$task" --input bf \
        --binarize \
        --init_from "$BF_CKPT" \
        --output "$out" \
        $EXTRA_ARGS
done

# ──────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Plotting control vs BF->GFP comparison ==="
python plot_control_vs_bfgfp.py \
    --results_dir "$LOO_OUT" \
    --output "$OUT_DIR/control_vs_bfgfp.png"

echo ""
echo "Done. Plot: $OUT_DIR/control_vs_bfgfp.png"

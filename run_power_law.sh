#!/bin/bash
# Train unet_2d_imagenet_pearson at multiple data fractions, evaluate, and plot.
#
# Usage:
#   bash run_power_law.sh
#
# Outputs:
#   ckpts/unet_2d_imagenet_pearson_frac{001,025,050,075,100}/  — checkpoints
#   results/power_law/frac{001,025,050,075,100}.json           — eval results
#   results/power_law/power_law.png                            — plot

set -e

CONFIG="configs/unet_2d_imagenet_pearson.yaml"
FRACTIONS="0.01 0.25 0.50 0.75 1.0"
RESULTS_DIR="results/power_law"

# Existing 100% model — reuse instead of retraining
EXISTING_CKPT="ckpts/unet_2d_imagenet_pearson/latest.pth"

mkdir -p "$RESULTS_DIR"

# ── Train each fraction ─────────────────────────────────────────
for FRAC in $FRACTIONS; do
    FRAC_TAG=$(printf "frac%03d" "$(echo "$FRAC * 100" | bc | cut -d. -f1)")
    CKPT_DIR="ckpts/unet_2d_imagenet_pearson_${FRAC_TAG}"

    # For 100%, symlink the existing checkpoint instead of retraining
    if [ "$FRAC" = "1.0" ]; then
        if [ ! -d "$CKPT_DIR" ]; then
            mkdir -p "$CKPT_DIR"
            ln -sf "$(realpath "$EXISTING_CKPT")" "${CKPT_DIR}/best.pth"
            echo "=== Linked existing 100% checkpoint to ${CKPT_DIR}/best.pth ==="
        else
            echo "=== Skipping fraction 1.0 (${CKPT_DIR} exists) ==="
        fi
        continue
    fi

    if [ -f "${CKPT_DIR}/best.pth" ]; then
        echo "=== Skipping fraction ${FRAC} (${CKPT_DIR}/best.pth exists) ==="
    else
        echo "=== Training fraction ${FRAC} ==="
        python train_fraction.py -c "$CONFIG" --fraction "$FRAC"
    fi
done

# ── Evaluate each fraction ──────────────────────────────────────
for FRAC in $FRACTIONS; do
    FRAC_TAG=$(printf "frac%03d" "$(echo "$FRAC * 100" | bc | cut -d. -f1)")
    CKPT_DIR="ckpts/unet_2d_imagenet_pearson_${FRAC_TAG}"
    OUT_JSON="${RESULTS_DIR}/${FRAC_TAG}.json"

    if [ -f "$OUT_JSON" ]; then
        echo "=== Skipping eval for ${FRAC} (${OUT_JSON} exists) ==="
    else
        echo "=== Evaluating fraction ${FRAC} ==="
        python eval_masked_pearson.py \
            -c "$CONFIG" \
            --checkpoint "${CKPT_DIR}/best.pth" \
            --output "$OUT_JSON"
    fi
done

# ── Plot ────────────────────────────────────────────────────────
echo "=== Plotting power law ==="
python plot_power_law.py --results_dir "$RESULTS_DIR" --output "${RESULTS_DIR}/power_law.png"

echo "=== Done. Plot saved to ${RESULTS_DIR}/power_law.png ==="

#!/bin/bash
# Step 3 only: LOO fine-tuning (BF -> {Exercise, Perturbation}) from each
# BF->GFP encoder checkpoint produced by Step 2, plus a frac000 baseline
# with no encoder init.
#
# Usage: bash scripts/run_step3_loo.sh

set -e

METADATA=data_mapping_drew.csv
LOO_CFG=configs/gfp_classifier_3d.yaml
LOO_OUT=results/loo
OUT_DIR=results/classifier

EXTRA_ARGS="$@"

mkdir -p "$LOO_OUT" "$OUT_DIR"

echo ""
echo "########################################"
echo "# STEP 3: LOO Fine-tuning (per task)   #"
echo "########################################"

# 0% baseline: no init_from, encoder stays ImageNet-only
for task in exercise perturbation; do
    echo "=== frac000 LOO ($task, BF) ==="
    python train_loo_classifier.py \
        -c "$LOO_CFG" \
        --metadata "$METADATA" \
        --task "$task" --input bf \
        --output "$LOO_OUT/frac000_${task}_bf.json" \
        $EXTRA_ARGS
done

# Fine-tune from each BF->GFP frac* checkpoint (must match LOO_CFG dims)
for ckpt in ckpts/*frac*/best.pth; do
    [ -f "$ckpt" ] || continue
    dir_name=$(basename "$(dirname "$ckpt")")
    if [[ "$dir_name" == gfp_classifier* ]]; then
        continue
    fi
    if [[ "$dir_name" =~ (frac[0-9]+) ]]; then
        tag="${BASH_REMATCH[1]}"
    else
        tag="$dir_name"
    fi
    for task in exercise perturbation; do
        echo "=== $tag LOO ($task, BF, init=$ckpt) ==="
        python train_loo_classifier.py \
            -c "$LOO_CFG" \
            --metadata "$METADATA" \
            --task "$task" --input bf \
            --init_from "$ckpt" \
            --output "$LOO_OUT/${tag}_${task}_bf.json" \
            $EXTRA_ARGS
    done
done

echo "=== Plotting LOO scaling curve ==="
python plot_loo_scaling.py \
    --results_dir "$LOO_OUT" --input bf \
    --output "$OUT_DIR/loo_scaling_bf.png"

echo ""
echo "Step 3 complete! Results in $LOO_OUT/ and $OUT_DIR/"

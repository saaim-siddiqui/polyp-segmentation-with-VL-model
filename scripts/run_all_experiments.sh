#!/bin/bash
# ==============================================================================
# Run All Experiments for VL Polyp Segmentation
# ==============================================================================
#
# This script runs the complete experimental pipeline:
# 1. Vision-only baseline
# 2. Full VL model
# 3. Ablation experiments (shape, size, location, pathology)
# 4. Evaluation on SUN database
# 5. Evaluation on benchmark datasets (Kvasir-SEG, CVC-ClinicDB)
#
# Usage:
#   bash scripts/run_all_experiments.sh
#
# Requirements:
#   - Data prepared in ./data/ directory
#   - GPU with at least 8GB VRAM
#   - ~24-48 hours total runtime
#
# ==============================================================================

set -e  # Exit on error

# Configuration
DATA_ROOT="./data/sun"
KVASIR_PATH="./data/kvasir_seg"
CVC_PATH="./data/cvc_clinicdb"
CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./results"
EPOCHS=100
BATCH_SIZE=4

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $RESULTS_DIR

echo "=============================================="
echo "VL Polyp Segmentation - Full Experiment Suite"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data root:     $DATA_ROOT"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Checkpoints:   $CHECKPOINT_DIR"
echo "  Results:       $RESULTS_DIR"
echo ""

# ==============================================================================
# PHASE 1: Training
# ==============================================================================

echo "=============================================="
echo "PHASE 1: TRAINING"
echo "=============================================="

# 1.1 Vision-Only Baseline
echo ""
echo "[1/6] Training Vision-Only Baseline..."
echo "----------------------------------------------"
python -m src.train \
    --data_root $DATA_ROOT \
    --experiment vision_only \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE 

echo "Done Vision-Only Training."

# # 1.2 Full VL Model
# echo ""
# echo "[2/6] Training Full VL Model..."
# echo "----------------------------------------------"
# python -m src.train \
#     --data_root $DATA_ROOT \
#     --experiment full \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --output_dir $CHECKPOINT_DIR/full_vl

# # 1.3 Ablation: Shape Only
# echo ""
# echo "[3/6] Training Ablation: Shape Only..."
# echo "----------------------------------------------"
# python -m src.train \
#     --data_root $DATA_ROOT \
#     --experiment text_shape_only \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --output_dir $CHECKPOINT_DIR/ablations/shape_only

# # 1.4 Ablation: Size Only
# echo ""
# echo "[4/6] Training Ablation: Size Only..."
# echo "----------------------------------------------"
# python -m src.train \
#     --data_root $DATA_ROOT \
#     --experiment text_size_only \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --output_dir $CHECKPOINT_DIR/ablations/size_only

# # 1.5 Ablation: Location Only
# echo ""
# echo "[5/6] Training Ablation: Location Only..."
# echo "----------------------------------------------"
# python -m src.train \
#     --data_root $DATA_ROOT \
#     --experiment text_location_only \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --output_dir $CHECKPOINT_DIR/ablations/location_only

# # 1.6 Ablation: Pathology Only
# echo ""
# echo "[6/6] Training Ablation: Pathology Only..."
# echo "----------------------------------------------"
# python -m src.train \
#     --data_root $DATA_ROOT \
#     --experiment text_pathology_only \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --output_dir $CHECKPOINT_DIR/ablations/pathology_only

# echo ""
# echo "✓ Phase 1 Complete: All models trained"
# echo ""

# # ==============================================================================
# # PHASE 2: Evaluation on SUN Database
# # ==============================================================================

# echo "=============================================="
# echo "PHASE 2: EVALUATION ON SUN DATABASE"
# echo "=============================================="

# # 2.1 Full Evaluation with Uncertainty Analysis
# echo ""
# echo "[1/2] Running full evaluation with uncertainty analysis..."
# echo "----------------------------------------------"
# python -m src.evaluate \
#     --checkpoint $CHECKPOINT_DIR/full_vl/best.pt \
#     --vision_checkpoint $CHECKPOINT_DIR/vision_only/best.pt \
#     --data_root $DATA_ROOT \
#     --output_dir $RESULTS_DIR/sun_full

# # 2.2 Evaluate all ablations
# echo ""
# echo "[2/2] Evaluating ablation models..."
# echo "----------------------------------------------"

# for ablation in shape_only size_only location_only pathology_only; do
#     echo "  Evaluating: $ablation"
#     python -m src.evaluate \
#         --checkpoint $CHECKPOINT_DIR/ablations/$ablation/best.pt \
#         --data_root $DATA_ROOT \
#         --output_dir $RESULTS_DIR/ablations/$ablation
# done

# echo ""
# echo "✓ Phase 2 Complete: SUN evaluation done"
# echo ""

# # ==============================================================================
# # PHASE 3: Benchmark Evaluation
# # ==============================================================================

# # echo "=============================================="
# # echo "PHASE 3: BENCHMARK EVALUATION"
# # echo "=============================================="

# # # Check if benchmark datasets exist
# # if [ -d "$KVASIR_PATH" ] && [ -d "$CVC_PATH" ]; then
# #     echo ""
# #     echo "Running benchmark evaluation..."
# #     echo "----------------------------------------------"
# #     python -m src.evaluate_benchmarks \
# #         --checkpoint $CHECKPOINT_DIR/full_vl/best.pt \
# #         --vision_checkpoint $CHECKPOINT_DIR/vision_only/best.pt \
# #         --kvasir_path $KVASIR_PATH \
# #         --cvc_path $CVC_PATH \
# #         --output_dir $RESULTS_DIR/benchmarks
# # else
# #     echo ""
# #     echo "⚠ Benchmark datasets not found. Skipping benchmark evaluation."
# #     echo "  To run benchmarks, download:"
# #     echo "  - Kvasir-SEG to $KVASIR_PATH"
# #     echo "  - CVC-ClinicDB to $CVC_PATH"
# # fi

# # echo ""
# # echo "✓ Phase 3 Complete"
# # echo ""

# # ==============================================================================
# # PHASE 4: Generate Summary
# # ==============================================================================

# echo "=============================================="
# echo "PHASE 4: GENERATE SUMMARY"
# echo "=============================================="

# echo ""
# echo "Generating results summary..."
# python -c "
# import json
# from pathlib import Path

# results_dir = Path('$RESULTS_DIR')

# print('\\n' + '='*60)
# print('EXPERIMENT RESULTS SUMMARY')
# print('='*60)

# # Load main results
# main_results = results_dir / 'sun_full' / 'evaluation_results.json'
# if main_results.exists():
#     with open(main_results) as f:
#         results = json.load(f)
    
#     print('\\nMain Results (SUN Database):')
#     print('-'*40)
    
#     if 'vl_model' in results:
#         print(f\"  VL Model Dice:     {results['vl_model']['dice']['mean']:.4f}\")
#         print(f\"  VL Model IoU:      {results['vl_model']['iou']['mean']:.4f}\")
    
#     if 'vision_only' in results:
#         print(f\"  Vision-Only Dice:  {results['vision_only']['dice']['mean']:.4f}\")
#         print(f\"  Vision-Only IoU:   {results['vision_only']['iou']['mean']:.4f}\")
    
#     if 'improvement' in results:
#         print(f\"\\n  Improvement (Dice): +{results['improvement']['dice']:.4f}\")
#         print(f\"  Improvement (IoU):  +{results['improvement']['iou']:.4f}\")

# print('\\n' + '='*60)
# print('Results saved to: $RESULTS_DIR/')
# print('='*60)
# "

# echo ""
# echo "=============================================="
# echo "ALL EXPERIMENTS COMPLETE!"
# echo "=============================================="
# echo ""
# echo "Results location: $RESULTS_DIR/"
# echo ""
# echo "Generated outputs:"
# echo "  - $RESULTS_DIR/sun_full/evaluation_results.json"
# echo "  - $RESULTS_DIR/sun_full/plots/"
# echo "  - $RESULTS_DIR/sun_full/uncertainty_maps/"
# echo "  - $RESULTS_DIR/ablations/"
# echo "  - $RESULTS_DIR/benchmarks/ (if benchmark data available)"
# echo ""
# echo "Next steps:"
# echo "  1. Review results in $RESULTS_DIR/"
# echo "  2. Generate paper figures: python scripts/generate_figures.py"
# echo "  3. Write your paper!"
# echo ""

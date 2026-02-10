#!/bin/bash
# run_everything.sh
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_OFFLINE=1


echo "Step 1: Running full metrics for pixelmask_cfg4..."
python evaluate_metrics.py \
  --json_path dataset_qwen_pe_top1000_captioned.json \
  --pred_dir pico_test/qwen_results_wosub_top1000_pixelmask_cfg4 \
  --gt_base_dir /host/ssd2/xiong-p/qwenpe/pico-banana-400k-subject_driven/openimages \
  --output_csv pico_test/qwen_results_wosub_top1000_pixelmask_cfg4/evaluation_results.csv \
  --gpu_ids 0,1,2,3,4,5,6,7

echo "Step 2: Running batch SigLIP-T updates for all models..."
bash run_all_evaluations.sh

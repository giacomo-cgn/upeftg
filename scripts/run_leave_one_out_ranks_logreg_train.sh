#!/usr/bin/env bash

feature_file=/leonardo_work/IscrC_cl-tta/gcignoni/upeftg/feature_bundles/raw_feature_bundle
manifest_dir=/leonardo_work/IscrC_cl-tta/gcignoni/upeftg/manifests/current/leave_one_out_ranks
output_root=/leonardo_work/IscrC_cl-tta/gcignoni/upeftg/runs/leave_one_out_ranks
features=(
  energy kurtosis l1_norm l2_norm concentration_of_energy
  sv_topk stable_rank spectral_entropy effective_rank
)

for rank in 8 16 32 64 128 256 512 1024 2048; do
  run_id="holdout_rank${rank}"

  python -m upeftguard.cli experiment train \
    --backend local \
    --manifest-json "$manifest_dir/${run_id}_train.json" \
    --feature-file "$feature_file" \
    --features "${features[@]}" \
    --model logistic_regression \
    --spectral-attention-granularity head \
    --spectral-moment-source sv \
    --spectral-qv-sum-mode append \
    --cv-strategy stratified \
    --output-root "$output_root" \
    --run-id "$run_id"
done

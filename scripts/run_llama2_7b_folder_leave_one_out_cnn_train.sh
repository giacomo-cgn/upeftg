#!/usr/bin/env bash

feature_file=/leonardo_work/IscrC_cl-tta/gcignoni/upeftg/feature_bundles/layer_sequence_feature_bundle/spectral_features.npy 
manifest_dir=/leonardo_work/IscrC_cl-tta/gcignoni/upeftg/manifests/current/llama2_7b_folder_leave_one_out
output_root=/leonardo_work/IscrC_cl-tta/gcignoni/upeftg/runs/llama2_7b_folder_leave_one_out_cnn
# features=(
#   energy kurtosis l1_norm l2_norm concentration_of_energy
#   sv_topk stable_rank spectral_entropy effective_rank
# )

cases=(
  holdout_alpaca
  holdout_hard_rank256
  holdout_ag_news_ripple
  holdout_ag_news_insertsent
  holdout_ag_news_syntactic
  holdout_ag_news_stybkd
)

for run_id in "${cases[@]}"; do
  python -m upeftguard.cli experiment train \
      --backend local \
      --manifest-json "$manifest_dir/${run_id}_train.json" \
      --feature-file "$feature_file" \
      --model cnn_1d \
      --hyperparams /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/manifests/cnn_hyperparams/cnn_1d_better1.json \
      --spectral-attention-granularity head \
      --spectral-moment-source sv \
      --spectral-qv-sum-mode append \
      --cv-strategy stratified \
      --output-root "$output_root" \
      --run-id "$run_id"
done

#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

stage=${1:-train}
if [[ "$stage" != train && "$stage" != infer ]]; then
  echo "Usage: $0 [train|infer]" >&2
  exit 2
fi

feature_file=feature_bundles/llama2_7b
manifest_dir=manifests/current/llama2_7b_folder_leave_one_out
output_root=runs/llama2_7b_folder_leave_one_out
features=(
  energy kurtosis l1_norm l2_norm concentration_of_energy
  sv_topk stable_rank spectral_entropy effective_rank
)

cases=(
  holdout_alpaca
  holdout_hard_rank256
  holdout_ag_news_ripple
  holdout_ag_news_insertsent
  holdout_ag_news_syntactic
  holdout_ag_news_stybkd
)

for run_id in "${cases[@]}"; do
  if [[ "$stage" == train ]]; then
    python -m upeftguard.cli experiment train \
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
  else
    python -m upeftguard.cli experiment infer \
      --checkpoint "$output_root/$run_id/training/models/best_model.joblib" \
      --manifest-json "$manifest_dir/${run_id}_infer.json" \
      --feature-file "$feature_file" \
      --output-root "$output_root" \
      --run-id "$run_id"
  fi
done

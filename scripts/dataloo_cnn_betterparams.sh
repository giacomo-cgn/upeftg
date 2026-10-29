#!/bin/bash

python -m upeftguard.cli experiment train \
  --backend local \
  --partition extra \
  --nodes auto \
  --workers-per-node auto \
  --cpus-per-worker 4 \
  --output-root /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/runs \
  --manifest-json /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/manifests/leave_one_dataset_out/non_imdb_plus_10_imdb_train.json \
  --feature-file /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/feature_bundles/layer_sequence_feature_bundle/spectral_features.npy \
  --model cnn_1d \
  --hyperparams /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/manifests/cnn_hyperparams/cnn_1d_better1.json \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode append \
  --spectral-attention-granularity head \
  --cv-strategy dataset_leave_one_out \
  --cv-seeds 42 \
  --random-state 42 \
  --no-split-by-folder \
  --input-normalization none \
  --n-jobs 4 \
  --task-mode binary \
  --selection-metric roc_auc \
  --no-refit \
  --run-id cnn_betterparams1
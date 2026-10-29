#!/bin/bash

  python -m upeftguard.cli experiment infer \
  --backend local \
  --partition extra \
  --checkpoint /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/runs/cnn_betterparams1/training/models/best_model.pt \
  --manifest-json /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/manifests/leave_one_dataset_out/imdb_minus_10_infer.json \
  --feature-file /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/feature_bundles/layer_sequence_feature_bundle/spectral_features.npy \
  --output-root /leonardo_work/IscrC_cl-tta/gcignoni/upeftg/runs \
  --run-id cnn_betterparams1_infer
#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi
cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

# CONDA_SH=${CONDA_SH:-/home/n.pitzalis/miniconda3/etc/profile.d/conda.sh}
# CONDA_ENV=${CONDA_ENV:-upeftg}
# if [[ -f "${CONDA_SH}" ]]; then
#   source "${CONDA_SH}"
#   conda activate "${CONDA_ENV}"
# fi

CURRENT_USER=${USER:-$(id -un)}
PROJECT_STORAGE_ROOT=${UPEFTGUARD_STORAGE_ROOT:-/models/${CURRENT_USER}/unsupervised-peftguard}
DATASET_ROOT=${DATASET_ROOT:-${UPEFTGUARD_DATA_ROOT:-${PROJECT_STORAGE_ROOT}/data}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/runs}
N_JOBS=${N_JOBS:--1}
DRY_RUN=${DRY_RUN:-0}
CLASS_WEIGHT_LOSS=${CLASS_WEIGHT_LOSS:-0}
RANK_LABEL_WEIGHT_LOSS=${RANK_LABEL_WEIGHT_LOSS:-0}

RUN_ID=${RUN_ID:-autoenceder_pretr_clean_squad_insertsent}
MANIFEST_JSON=${MANIFEST_JSON:-${REPO_ROOT}/manifests/single_datasets/llama2_7b_squad_insertsent.json}
FEATURE_FILE=${FEATURE_FILE:-${REPO_ROOT}/runs/feature_extract/list2_features-merged-cnn/merged/spectral_features.npy}
AUTOENCODER_HYPERPARAMS=${AUTOENCODER_HYPERPARAMS:-${REPO_ROOT}/manifests/selfsupervised_hyperparams/autoencoder_pretrained_hyperparams.json}
FEATURES=(
  energy
  kurtosis
  l1_norm
  l2_norm
  linf_norm
  mean_abs
  concentration_of_energy
  sv_topk
  stable_rank
  spectral_entropy
  effective_rank
)

COMMON_ARGS=(
  python -m upeftguard.cli run supervised
  --feature-file "${FEATURE_FILE}"
  --features "${FEATURES[@]}"
  --model autoencoder_knn
  --autoencoder-hyperparams "${AUTOENCODER_HYPERPARAMS}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --train-split 80
  --calibration-split 20
  --accepted-fpr 0.01 0.05 0.1
  --split-by-folder
  --cv-folds 5
  --cv-seeds 42
  --random-state 42
  --spectral-sv-top-k 8
  --spectral-moment-source both
  --spectral-qv-sum-mode append
  --spectral-entrywise-delta-mode dense
  --skip-feature-importance
  --n-jobs "${N_JOBS}"
)

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi
if [[ "${CLASS_WEIGHT_LOSS}" == "1" || "${CLASS_WEIGHT_LOSS,,}" == "true" ]]; then
  COMMON_ARGS+=(--class-weight-loss)
fi
if [[ "${RANK_LABEL_WEIGHT_LOSS}" == "1" || "${RANK_LABEL_WEIGHT_LOSS,,}" == "true" ]]; then
  COMMON_ARGS+=(--rank-label-weight-loss)
fi

echo "Submitting ${RUN_ID} from ${MANIFEST_JSON}"
echo "Note: rank512 label0_2 is excluded because it is absent from the current feature bundle."
"${COMMON_ARGS[@]}" \
  --manifest-json "${MANIFEST_JSON}" \
  --run-id "${RUN_ID}"

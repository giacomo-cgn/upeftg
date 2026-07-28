# upeftguard

Spectral PEFT-adapter feature extraction and supervised backdoor detection.

The public workflow is model-neutral. It supports classical scikit-learn
estimators, CNN, DANN, and Transformer models through one CLI:

```bash
python -m upeftguard.cli experiment <extract|aggregate|train|infer|explain|full>
```

## Environment

Create the Conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate upeftg
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
conda activate upeftg
```

Run commands from the repository root.

## End-to-end experiment pipeline

The commands below start with an empty data directory and produce the raw
spectral feature bundles, trained logistic-regression checkpoints, and
inference reports used by the current experiments. They use `--backend local`,
which requires no cluster and runs each stage in the current process. If Slurm
is available, replace it with `--backend slurm`; the CLI will submit the work
to the cluster instead.

### 1. Download PADBench

The adapters are hosted in the
[PADBench dataset](https://huggingface.co/datasets/Vincent-HKUSTGZ/PADBench).
Create a dataset directory. Any local path works with the local backend; for
Slurm, choose a shared path visible from every compute node:

```bash
export UPEFTGUARD_DATA_ROOT=/path/to/shared/padbench
mkdir -p "$UPEFTGUARD_DATA_ROOT"
```

Download PADBench into that directory. Its adapter folders, such as
`llama2_7b_ag_news_insertsent_rank256_qv`, must be directly beneath
`$UPEFTGUARD_DATA_ROOT`.

### 2. Extract spectral features

The main feature bundle contains all Llama-2-7B adapters needed by the
Llama-2-7B in-domain, AG News zero-shot, AG News leave-one-attack-out, and
folder leave-one-out experiments:

```bash
ARCH=llama2_7b
EXTRACTION_MANIFEST=manifests/llama2_7b_feature_extraction.json

python -m upeftguard.cli experiment extract \
  --backend local \
  --manifest-json "$EXTRACTION_MANIFEST" \
  --dataset-root "$UPEFTGUARD_DATA_ROOT" \
  --output-root runs/feature_extraction \
  --run-id "$ARCH" \
  --features energy kurtosis l1_norm l2_norm linf_norm mean_abs concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode append \
  --spectral-attention-granularity head
```

With the local backend, feature extraction is complete when the command
returns. Its output is
`runs/feature_extraction/llama2_7b/extraction/features/spectral_features.npy`.

With the Slurm backend, submission is asynchronous, so wait for the dependent
merge job to produce that file before training. Slurm users can also add
`--partition`, `--cpus-per-worker`, `--nodes`, `--workers-per-node`, and
rank-specific allocation flags. See
[Slurm feature-extraction orchestration](docs/slurm-orchestration.md#feature-extraction-orchestration)
for resource discovery, rank-specific worker settings, job dependencies, logs,
and dry runs.

The in-domain architecture experiments require one feature bundle per
architecture. Reuse the command above with the corresponding `ARCH` and
`EXTRACTION_MANIFEST`:

| `ARCH` | `EXTRACTION_MANIFEST` |
| --- | --- |
| `llama2_7b` | `manifests/llama2_7b_feature_extraction.json` |
| `llama2_13b` | `manifests/architecture_exploration/llama2_13b_architecture_tbh.json` |
| `llama3_8b` | `manifests/architecture_exploration/llama3_8b_architecture_tbh.json` |
| `qwen1.5_7b` | `manifests/architecture_exploration/qwen1.5_7b_architecture_tbh.json` |
| `roberta_base` | `manifests/architecture_exploration/roberta_base_architecture.json` |
| `flan_t5_xl` | `manifests/architecture_exploration/flan_t5_xl_architecture_tbh.json` |

Each completed bundle is at
`runs/feature_extraction/<ARCH>/extraction/features/spectral_features.npy`.

### 3. Train

All five experiment families use the same training command. First select the
paired manifests, feature bundle, output root, and run ID from the
[experiment manifest catalogue](#5-experiment-manifest-catalogue), then run:

```bash
python -m upeftguard.cli experiment train \
  --backend local \
  --manifest-json "$TRAIN_MANIFEST" \
  --dataset-root "$UPEFTGUARD_DATA_ROOT" \
  --feature-file "$FEATURE_FILE" \
  --features energy kurtosis l1_norm l2_norm concentration_of_energy sv_topk stable_rank spectral_entropy effective_rank \
  --spectral-sv-top-k 8 \
  --spectral-moment-source sv \
  --spectral-qv-sum-mode append \
  --spectral-attention-granularity head \
  --model logistic_regression \
  --cv-folds 5 \
  --cv-strategy stratified \
  --input-normalization none \
  --random-state 42 \
  --n-jobs -1 \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID"
```

With the local backend, the checkpoint is ready when training returns. With the
Slurm backend, wait for
`$OUTPUT_ROOT/$RUN_ID/training/models/best_model.joblib` before inference.
Classical models use their registered candidate grid when `--hyperparams` is
omitted. CNN, DANN, and Transformer models require a `--hyperparams` manifest;
see the
[model hyperparameter documentation](docs/experiment-runbook.md#model-hyperparameters).

### 4. Run inference

Use the inference manifest paired with the training manifest and the same
feature bundle, output root, and run ID:

```bash
python -m upeftguard.cli experiment infer \
  --backend local \
  --checkpoint "$OUTPUT_ROOT/$RUN_ID/training/models/best_model.joblib" \
  --manifest-json "$INFER_MANIFEST" \
  --feature-file "$FEATURE_FILE" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID"
```

The final metrics are written to
`$OUTPUT_ROOT/$RUN_ID/inference/reports/report.json`.

### 5. Experiment manifest catalogue

The training and inference commands are shared. Set the five variables shown
for the desired experiment, then run the commands in sections
[3](#3-train) and [4](#4-run-inference).

#### Llama-2-7B in-domain

This experiment uses the Llama-2-7B pair in `architecture_classifiers`:

```bash
TRAIN_MANIFEST=manifests/architecture_classifiers/llama2_7b_train.json
INFER_MANIFEST=manifests/architecture_classifiers/llama2_7b_infer.json
FEATURE_FILE=runs/feature_extraction/llama2_7b/extraction/features/spectral_features.npy
OUTPUT_ROOT=runs/architecture_classifiers
RUN_ID=llama2_7b
```

#### AG News attack zero-shot

Choose distinct source and target attacks from `insertsent`, `ripple`,
`stybkd`, and `syntactic`. Every ordered source-to-target pair is present in
[`manifests/zero_shots_ag_news`](manifests/zero_shots_ag_news/). Set `VARIANT`
to an empty string for the standard split or `_clean50_50` for that manifest
variant:

```bash
SOURCE=insertsent
TARGET=ripple
VARIANT=

TRAIN_MANIFEST="manifests/zero_shots_ag_news/llama2_7b_ag_news_zero_shot_${SOURCE}_to_${TARGET}${VARIANT}_train.json"
INFER_MANIFEST="manifests/zero_shots_ag_news/llama2_7b_ag_news_zero_shot_${SOURCE}_to_${TARGET}${VARIANT}_infer.json"
FEATURE_FILE=runs/feature_extraction/llama2_7b/extraction/features/spectral_features.npy
OUTPUT_ROOT=runs/zero_shots_ag_news
RUN_ID="${SOURCE}_to_${TARGET}${VARIANT}"
```

#### AG News leave-one-attack-out

There are four paired manifests in
[`manifests/loo_ag_news_attacks`](manifests/loo_ag_news_attacks/). Set
`MANIFEST_ATTACK` to `insertsent`, `RIPPLE`, `stybkd`, or `syntactic`:

```bash
MANIFEST_ATTACK=RIPPLE

TRAIN_MANIFEST="manifests/loo_ag_news_attacks/holdout_llama2_7b_ag_news_${MANIFEST_ATTACK}_rank256_qv_train.json"
INFER_MANIFEST="manifests/loo_ag_news_attacks/holdout_llama2_7b_ag_news_${MANIFEST_ATTACK}_rank256_qv_infer.json"
FEATURE_FILE=runs/feature_extraction/llama2_7b/extraction/features/spectral_features.npy
OUTPUT_ROOT=runs/loo_ag_news_attacks
RUN_ID="holdout_${MANIFEST_ATTACK}"
```

#### Llama-2-7B folder leave-one-out

Set `CASE` to one of `holdout_alpaca`, `holdout_hard_rank256`,
`holdout_ag_news_ripple`, `holdout_ag_news_insertsent`,
`holdout_ag_news_syntactic`, or `holdout_ag_news_stybkd`. The paired manifests
are in
[`manifests/llama2_7b_folder_leave_one_out`](manifests/llama2_7b_folder_leave_one_out/):

```bash
CASE=holdout_alpaca

TRAIN_MANIFEST="manifests/llama2_7b_folder_leave_one_out/${CASE}_train.json"
INFER_MANIFEST="manifests/llama2_7b_folder_leave_one_out/${CASE}_infer.json"
FEATURE_FILE=runs/feature_extraction/llama2_7b/extraction/features/spectral_features.npy
OUTPUT_ROOT=runs/llama2_7b_folder_leave_one_out
RUN_ID="$CASE"
```

#### In-domain on different architectures

Set `ARCH` to `llama2_7b`, `llama2_13b`, `llama3_8b`, `qwen1.5_7b`,
`roberta_base`, or `flan_t5_xl`. The train/inference split for each
architecture is in
[`manifests/architecture_classifiers`](manifests/architecture_classifiers/):

```bash
ARCH=llama2_13b

TRAIN_MANIFEST="manifests/architecture_classifiers/${ARCH}_train.json"
INFER_MANIFEST="manifests/architecture_classifiers/${ARCH}_infer.json"
FEATURE_FILE="runs/feature_extraction/${ARCH}/extraction/features/spectral_features.npy"
OUTPUT_ROOT=runs/architecture_classifiers
RUN_ID="$ARCH"
```

## Public Workflows

| Command | Purpose |
| --- | --- |
| `experiment extract` | Extract spectral features from adapters in a manifest. |
| `experiment aggregate` | Convert extracted features into the layer-sequence representation used by CNN, DANN, and Transformer models. |
| `experiment train` | Train and finalize a selected supervised model. Sequence models consume aggregated features; classical models can consume extracted tabular features directly. |
| `experiment infer` | Evaluate a self-contained `.pt` or `.joblib` checkpoint against an explicit inference manifest and compatible feature artifact. |
| `experiment explain` | Submit grouped linear SHAP explanations for a binary logistic-regression checkpoint. |
| `experiment full` | Run extraction, optional aggregation, training, and checkpoint inference as one local or Slurm workflow. |

The authoritative run root is `runs/<run-id>/`. Each selected stage owns its
inputs, operational state, logs, scientific artifacts, and reports beneath that
root.

## Documentation

- [Architecture](docs/architecture.md): package boundaries, dependency direction, public APIs, and runtime-data policy.
- [Pipeline overview](docs/pipeline-overview.md): end-to-end data flow, artifact representations, supervised lifecycle, and output layout.
- [Experiment CLI runbook](docs/experiment-runbook.md): complete commands, flag defaults, manifest forms, cross-validation, calibration, and experiment variants.
- [Spectral features](docs/spectral-features.md): feature definitions, numerical implementation, spectral flags, schemas, and metadata.
- [Slurm orchestration](docs/slurm-orchestration.md): job graphs, resource discovery, packed workers, dependencies, logs, dry runs, and troubleshooting.


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import joblib
except Exception:  # pragma: no cover - fallback for minimal environments
    joblib = None
import pickle

from ..features.spectral import (
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    build_spectral_feature_names,
    resolve_spectral_features,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
    sanitize_spectral_metadata,
    spectral_block_lora_dims_by_block,
    spectral_extractor_params,
)
from ..unsupervised.reporting import (
    compute_infer_threshold_rows,
    compute_infer_threshold_rows_from_inference,
    compute_offline_metrics,
    save_score_csv,
    summarize_scores,
)
from ..utilities.artifacts.export_winner_feature_weights import (
    cleanup_winner_feature_weights_export,
    export_winner_feature_weights,
    merge_winner_feature_weights_export,
    prepare_winner_feature_weights_export,
    run_winner_feature_weights_export_worker,
)
from ..utilities.artifacts.spectral_metadata import (
    dataset_layouts_from_source,
    load_spectral_metadata,
    resolve_dataset_reference_for_metadata,
    write_spectral_metadata,
)
from ..utilities.core.manifest import (
    AttackSampleIdentity,
    infer_attack_sample_identities,
    parse_joint_manifest_json,
    parse_joint_manifest_json_by_model_name,
    parse_single_manifest_json,
    parse_single_manifest_json_by_model_name,
    resolve_manifest_path,
)
from ..utilities.core.run_context import RunContext, create_run_context
from ..utilities.core.serialization import json_ready
from ..utilities.merge.merge_feature_files import (
    DEFAULT_FEATURE_EXTRACT_ROOT,
    _resolve_existing_companion_path,
    _resolve_feature_extract_root,
    _resolve_input_feature_path,
)
from .distributed import (
    build_slurm_array_next_steps,
    resolve_slurm_cpus_per_task,
    resolve_slurm_max_concurrent,
)
from .interfaces import (
    ATTACK_FAMILY_MULTICLASS_ATTACKS,
    ARCHITECTURE_INDEPENDENT_AGGREGATION_KIND,
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
    SUPERVISED_TASK_MODE_BINARY,
    SupervisedFeatureBundle,
    SupervisedPredictionOutputs,
    SupervisedTaskSpec,
    TABULAR_SPECTRAL_REPRESENTATION_KIND,
)
from .registry import (
    CNN_1D_DANN_MODEL_NAME,
    CNN_1D_MODEL_NAME,
    candidate_params,
    create,
    model_backend,
    model_complexity_rank,
    normalization_policy,
    registered_models,
    resolve_autoencoder_hyperparams,
    resolve_cnn_hyperparams,
    supported_representation_kinds,
)


SCRIPT_VERSION = "1.0.0"
PIPELINE_NAME = "supervised"
FINALIZE_STATE_FILENAME = "finalize_state.json"
FINALIZE_EXPORT_TRAIN_FEATURES_FILENAME = "winner_feature_weights_train_features.npy"
FINALIZE_EXPORT_TRAIN_LABELS_FILENAME = "winner_feature_weights_train_labels.npy"
SELECTED_THRESHOLD_FILENAME = "selected_threshold.json"
OPEN_SET_UNKNOWN_ATTACK_NAME = "unknown_attack"
OPEN_SET_ATTACK_FPR = 0.05
OPEN_SET_KNOWN_ATTACK_MISS_RATE = 0.05
GROUP_MASK_SUFFIX = "_group_mask.npy"
VALUE_MASK_SUFFIX = "_value_mask.npy"
GROUP_NAMES_SUFFIX = "_group_names.json"
DANN_DEFAULT_SOURCE_RANK = 256
DANN_DEFAULT_LAMBDA_MAX = 1.0
DANN_DEFAULT_LAMBDA_GAMMA = 10.0
DANN_DEFAULT_LR_ALPHA = 10.0
DANN_DEFAULT_LR_BETA = 0.75
DANN_DEFAULT_TARGET_ADAPTATION_PERCENT = 80
SELECTION_METRIC_TASK_DEFAULT = "task_default"
SELECTION_METRIC_ROC_AUC = "roc_auc"
SELECTION_METRIC_MACRO_F1 = "macro_f1"
SELECTION_METRIC_BINARY_AUROC = "binary_auroc"
SUPPORTED_SELECTION_METRICS = (
    SELECTION_METRIC_TASK_DEFAULT,
    SELECTION_METRIC_ROC_AUC,
    SELECTION_METRIC_MACRO_F1,
    SELECTION_METRIC_BINARY_AUROC,
)
_RANK_TOKEN_RE = re.compile(r"(?:^|_)rank(\d+(?:\.\d+)?)(?:_|$)", re.IGNORECASE)


def _default_binary_task_spec() -> SupervisedTaskSpec:
    class_names = ("clean", "backdoored")
    return SupervisedTaskSpec(
        task_mode=SUPERVISED_TASK_MODE_BINARY,
        class_names=class_names,
        class_to_index={name: idx for idx, name in enumerate(class_names)},
        binary_projection=BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    )


def _resolve_attack_family_multiclass_task_spec(
    attack_names: list[str] | tuple[str, ...] | None,
) -> SupervisedTaskSpec:
    if not attack_names:
        raise ValueError(
            "task_mode=attack_family_multiclass requires --multiclass-attack-names with at least "
            "one positive attack class name"
        )

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in attack_names:
        name = str(raw_name).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        normalized_names.append(name)

    if not normalized_names:
        raise ValueError(
            "task_mode=attack_family_multiclass requires at least one non-empty attack class name"
        )

    reserved_names = {"clean", OPEN_SET_UNKNOWN_ATTACK_NAME}
    conflicting_names = [
        name for name in normalized_names if str(name).strip().lower() in reserved_names
    ]
    if conflicting_names:
        raise ValueError(
            "attack_family_multiclass class names cannot reuse reserved labels "
            f"{sorted(reserved_names)}; got {conflicting_names}"
        )

    class_names = ("clean", *normalized_names)
    return SupervisedTaskSpec(
        task_mode=SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS,
        class_names=class_names,
        class_to_index={name: idx for idx, name in enumerate(class_names)},
        binary_projection=BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    )


def _resolve_supervised_task_spec(
    *,
    task_mode: str | None,
    multiclass_attack_names: list[str] | tuple[str, ...] | None,
) -> SupervisedTaskSpec:
    resolved_task_mode = str(task_mode or SUPERVISED_TASK_MODE_BINARY)
    if resolved_task_mode == SUPERVISED_TASK_MODE_BINARY:
        return _default_binary_task_spec()
    if resolved_task_mode == SUPERVISED_TASK_MODE_ATTACK_FAMILY_MULTICLASS:
        return _resolve_attack_family_multiclass_task_spec(multiclass_attack_names)
    raise ValueError(f"Unsupported supervised task_mode={resolved_task_mode!r}")


def _resolve_selection_metric(
    selection_metric: str | None,
    *,
    task_spec: SupervisedTaskSpec,
) -> str:
    raw_metric = str(selection_metric or SELECTION_METRIC_TASK_DEFAULT).strip().lower()
    if raw_metric in {"", "default", SELECTION_METRIC_TASK_DEFAULT}:
        return str(task_spec.selection_metric_name)
    if raw_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported supervised selection metric {selection_metric!r}; expected one of "
            f"{list(SUPPORTED_SELECTION_METRICS)}"
        )
    if raw_metric == SELECTION_METRIC_MACRO_F1 and task_spec.is_binary:
        raise ValueError("--selection-metric macro_f1 is only supported for multiclass supervised tasks")
    if raw_metric == SELECTION_METRIC_ROC_AUC:
        return SELECTION_METRIC_ROC_AUC if task_spec.is_binary else SELECTION_METRIC_BINARY_AUROC
    return raw_metric


def _task_spec_from_payload(payload: Any) -> SupervisedTaskSpec:
    if not isinstance(payload, dict):
        return _default_binary_task_spec()

    task_mode = str(payload.get("task_mode") or SUPERVISED_TASK_MODE_BINARY)
    class_names_raw = payload.get("class_names")
    if not isinstance(class_names_raw, list) or not class_names_raw:
        return _default_binary_task_spec()
    class_names = tuple(str(x) for x in class_names_raw)

    class_to_index_raw = payload.get("class_to_index")
    if isinstance(class_to_index_raw, dict) and class_to_index_raw:
        class_to_index = {str(key): int(value) for key, value in class_to_index_raw.items()}
    else:
        class_to_index = {name: idx for idx, name in enumerate(class_names)}

    binary_projection = str(
        payload.get(
            "binary_projection",
            (
                BINARY_PROJECTION_POSITIVE_CLASS_SCORE
                if task_mode == SUPERVISED_TASK_MODE_BINARY
                else BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY
            ),
        )
    )
    return SupervisedTaskSpec(
        task_mode=task_mode,
        class_names=class_names,
        class_to_index=class_to_index,
        binary_projection=binary_projection,
    )


def _task_spec_from_manifest(manifest: dict[str, Any]) -> SupervisedTaskSpec:
    return _task_spec_from_payload(manifest.get("task"))


@dataclass(frozen=True)
class ResolvedFeatureBundlePaths:
    feature_path: Path
    model_names_path: Path
    metadata_path: Path | None
    group_mask_path: Path | None
    value_mask_path: Path | None
    group_names_path: Path | None


def _detect_manifest_mode(manifest_json: Path) -> str:
    resolved_manifest_json = resolve_manifest_path(manifest_json)
    with open(resolved_manifest_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "train" in payload and "infer" in payload:
        return "joint"
    return "single"


def _labels_from_items(
    items: list[Any],
    *,
    task_spec: SupervisedTaskSpec,
    sample_identities: list[AttackSampleIdentity] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    if task_spec.is_binary:
        raw_labels = [item.label for item in items]
        values = np.asarray([int(label) if label is not None else -1 for label in raw_labels], dtype=np.int32)
        known = np.asarray([label is not None for label in raw_labels], dtype=bool)
        return values, known, raw_labels

    if sample_identities is None or len(sample_identities) != len(items):
        raise ValueError("Multiclass supervised label derivation requires aligned attack sample identities")

    raw_labels: list[int | None] = []
    for item, identity in zip(items, sample_identities):
        if item.label is None:
            raw_labels.append(None)
            continue
        if int(item.label) == 0:
            raw_labels.append(int(task_spec.clean_class_index))
            continue

        attack_name = str(identity.attack_name)
        if attack_name not in task_spec.class_to_index:
            raise ValueError(
                "task_mode=attack_family_multiclass encountered a positive sample outside the configured "
                f"attack vocabulary: model={item.model_name!r}, attack_name={attack_name!r}, "
                f"supported={list(task_spec.class_names[1:])}"
            )
        raw_labels.append(int(task_spec.class_to_index[attack_name]))

    values = np.asarray([int(label) if label is not None else -1 for label in raw_labels], dtype=np.int32)
    known = np.asarray([label is not None for label in raw_labels], dtype=bool)
    return values, known, raw_labels


def _resolve_train_split_percent(train_split_percent: int) -> int:
    resolved = int(train_split_percent)
    if resolved < 1 or resolved > 100:
        raise ValueError(f"train_split must be in the range [1, 100], got {resolved}")
    return resolved


def _resolve_calibration_split_percent(calibration_split_percent: int | None) -> int | None:
    if calibration_split_percent is None:
        return None
    resolved = int(calibration_split_percent)
    if resolved < 1 or resolved > 99:
        raise ValueError(f"calibration_split must be in the range [1, 99], got {resolved}")
    return resolved


def _resolve_accepted_fprs(
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    if accepted_fpr is None:
        return None

    raw_values: list[float]
    if isinstance(accepted_fpr, (list, tuple)):
        raw_values = [float(value) for value in accepted_fpr]
    else:
        raw_values = [float(accepted_fpr)]

    if not raw_values:
        raise ValueError("accepted_fpr must include at least one value when provided")

    resolved_values: list[float] = []
    seen_values: set[float] = set()
    for value in raw_values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"accepted_fpr must be in the range [0.0, 1.0], got {value}")
        if value in seen_values:
            continue
        seen_values.add(value)
        resolved_values.append(float(value))
    return resolved_values


def _resolve_supervised_feature_bundle_paths(
    feature_spec: Path,
    *,
    feature_root: Path = DEFAULT_FEATURE_EXTRACT_ROOT,
) -> ResolvedFeatureBundlePaths:
    resolved_feature_root = _resolve_feature_extract_root(feature_root)
    candidate = feature_spec.expanduser()
    local_candidate = candidate if candidate.is_absolute() else (Path.cwd().resolve() / candidate)
    resolved_local_candidate = local_candidate.resolve()

    if resolved_local_candidate.exists():
        resolved_feature_path = (
            resolved_local_candidate / "spectral_features.npy"
            if resolved_local_candidate.is_dir()
            else resolved_local_candidate
        )
    else:
        resolved_feature_path = _resolve_input_feature_path(
            feature_spec,
            feature_root=resolved_feature_root,
        )

    resolved_feature_path = resolved_feature_path.expanduser().resolve()
    if not resolved_feature_path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {resolved_feature_path}")

    resolved_model_names_path = _resolve_existing_companion_path(
        resolved_feature_path,
        "_model_names.json",
        required=True,
    ).expanduser().resolve()
    metadata_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        "_metadata.json",
        required=False,
    )
    group_mask_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        GROUP_MASK_SUFFIX,
        required=False,
    )
    value_mask_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        VALUE_MASK_SUFFIX,
        required=False,
    )
    group_names_candidate = _resolve_existing_companion_path(
        resolved_feature_path,
        GROUP_NAMES_SUFFIX,
        required=False,
    )
    return ResolvedFeatureBundlePaths(
        feature_path=resolved_feature_path,
        model_names_path=resolved_model_names_path,
        metadata_path=(metadata_candidate.expanduser().resolve() if metadata_candidate.exists() else None),
        group_mask_path=(
            group_mask_candidate.expanduser().resolve() if group_mask_candidate.exists() else None
        ),
        value_mask_path=(
            value_mask_candidate.expanduser().resolve() if value_mask_candidate.exists() else None
        ),
        group_names_path=(
            group_names_candidate.expanduser().resolve() if group_names_candidate.exists() else None
        ),
    )


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _resolve_holdout_subset_count(
    *,
    bucket_size: int,
    subset_percent: int,
    bucket_name: str,
    split_name: str,
    subset_label: str,
) -> tuple[int, bool]:
    if bucket_size < 2:
        raise ValueError(
            f"{split_name} requires at least two samples in each split bucket when creating a holdout split, "
            f"but '{bucket_name}' has {bucket_size}"
        )

    raw_subset = (int(bucket_size) * int(subset_percent)) / 100.0
    rounded_subset = _round_half_up(raw_subset)
    subset_count = min(max(rounded_subset, 1), int(bucket_size) - 1)
    return int(subset_count), bool(subset_count != rounded_subset)


def _label_count_rows(labels: np.ndarray) -> list[dict[str, int]]:
    if labels.size == 0:
        return []
    unique, counts = np.unique(labels, return_counts=True)
    return [
        {
            "label": int(label),
            "count": int(count),
        }
        for label, count in zip(unique.tolist(), counts.tolist())
    ]


def _named_label_count_rows(labels: np.ndarray, *, task_spec: SupervisedTaskSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _label_count_rows(labels):
        label_index = int(row["label"])
        class_name = (
            str(task_spec.class_names[label_index])
            if 0 <= label_index < task_spec.n_classes
            else f"unknown_{label_index}"
        )
        rows.append(
            {
                "label": label_index,
                "class_name": class_name,
                "count": int(row["count"]),
            }
        )
    return rows


def _project_optional_labels_to_binary(
    labels: list[int | None],
    *,
    task_spec: SupervisedTaskSpec,
) -> list[int | None]:
    return [task_spec.project_label_to_binary(label) for label in labels]


def _distribution_rows_from_predictions(
    predicted_labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
) -> list[dict[str, Any]]:
    predicted = np.asarray(predicted_labels, dtype=np.int32).reshape(-1)
    unique, counts = np.unique(predicted, return_counts=True)
    return [
        {
            "class_index": int(class_index),
            "class_name": str(task_spec.class_names[int(class_index)]),
            "count": int(count),
        }
        for class_index, count in zip(unique.tolist(), counts.tolist())
    ]


def _open_set_unknown_class_index(task_spec: SupervisedTaskSpec) -> int:
    return int(task_spec.n_classes)


def _open_set_class_name(class_index: int, *, task_spec: SupervisedTaskSpec) -> str:
    resolved = int(class_index)
    if 0 <= resolved < task_spec.n_classes:
        return str(task_spec.class_names[resolved])
    if resolved == _open_set_unknown_class_index(task_spec):
        return OPEN_SET_UNKNOWN_ATTACK_NAME
    return f"unknown_{resolved}"


def _csv_safe_class_name(class_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(class_name)).strip("_") or "class"


def _open_set_distribution_rows(
    predicted_labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
) -> list[dict[str, Any]]:
    predicted = np.asarray(predicted_labels, dtype=np.int32).reshape(-1)
    if predicted.size == 0:
        return []
    unique, counts = np.unique(predicted, return_counts=True)
    return [
        {
            "class_index": int(class_index),
            "class_name": _open_set_class_name(int(class_index), task_spec=task_spec),
            "count": int(count),
        }
        for class_index, count in zip(unique.tolist(), counts.tolist())
    ]


def _observed_known_attack_indices(labels: np.ndarray, *, task_spec: SupervisedTaskSpec) -> list[int]:
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    unique = sorted(int(x) for x in np.unique(labels_np).tolist())
    return [
        label
        for label in unique
        if label != task_spec.clean_class_index and 0 <= label < task_spec.n_classes
    ]


def _finite_or_default(values: np.ndarray, default: float) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values_np[np.isfinite(values_np)]
    if finite.size == 0:
        return np.asarray([float(default)], dtype=np.float64)
    return finite


def _build_open_set_unknown_attack_config(
    *,
    train_labels: np.ndarray,
    train_outputs: SupervisedPredictionOutputs,
    task_spec: SupervisedTaskSpec,
) -> dict[str, Any] | None:
    if task_spec.is_binary or train_outputs.probabilities is None:
        return None

    labels = np.asarray(train_labels, dtype=np.int32).reshape(-1)
    probabilities = np.asarray(train_outputs.probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != task_spec.n_classes:
        return None

    known_attack_indices = _observed_known_attack_indices(labels, task_spec=task_spec)
    if not known_attack_indices:
        return None

    scores = np.asarray(train_outputs.backdoor_scores, dtype=np.float64).reshape(-1)
    clean_scores = _finite_or_default(
        scores[labels == task_spec.clean_class_index],
        default=0.5,
    )
    attack_threshold = float(np.percentile(clean_scores, 100.0 * (1.0 - OPEN_SET_ATTACK_FPR)))

    true_known_confidences: list[float] = []
    known_set = set(int(x) for x in known_attack_indices)
    for row_idx, label in enumerate(labels.tolist()):
        label_index = int(label)
        if label_index not in known_set:
            continue
        true_known_confidences.append(float(probabilities[int(row_idx), label_index]))
    known_conf = _finite_or_default(np.asarray(true_known_confidences, dtype=np.float64), default=0.5)
    known_attack_confidence_threshold = float(
        np.percentile(known_conf, 100.0 * OPEN_SET_KNOWN_ATTACK_MISS_RATE)
    )

    return {
        "method": "backdoor_score_and_known_attack_confidence",
        "unknown_class_name": OPEN_SET_UNKNOWN_ATTACK_NAME,
        "unknown_class_index": _open_set_unknown_class_index(task_spec),
        "attack_threshold": attack_threshold,
        "attack_threshold_source": "train_clean_backdoor_score_p95",
        "attack_threshold_target_fpr": float(OPEN_SET_ATTACK_FPR),
        "known_attack_confidence_threshold": known_attack_confidence_threshold,
        "known_attack_confidence_threshold_source": "train_known_attack_true_class_probability_p05",
        "known_attack_confidence_target_miss_rate": float(OPEN_SET_KNOWN_ATTACK_MISS_RATE),
        "known_attack_indices": [int(x) for x in known_attack_indices],
        "known_attack_names": [str(task_spec.class_names[int(x)]) for x in known_attack_indices],
        "rule": (
            "clean if backdoor_score < attack_threshold; unknown_attack if backdoor_score >= "
            "attack_threshold and max probability over training-observed attack heads is below "
            "known_attack_confidence_threshold; otherwise argmax over training-observed attack heads"
        ),
    }


def _apply_open_set_unknown_attack_rule(
    *,
    outputs: SupervisedPredictionOutputs,
    task_spec: SupervisedTaskSpec,
    config: dict[str, Any] | None,
) -> dict[str, np.ndarray] | None:
    if task_spec.is_binary or config is None or outputs.probabilities is None:
        return None

    probabilities = np.asarray(outputs.probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != task_spec.n_classes:
        return None

    known_attack_indices = [int(x) for x in config.get("known_attack_indices", [])]
    if not known_attack_indices:
        return None

    scores = np.asarray(outputs.backdoor_scores, dtype=np.float64).reshape(-1)
    known_probs = probabilities[:, known_attack_indices]
    known_rel_argmax = np.argmax(known_probs, axis=1)
    known_confidence = np.max(known_probs, axis=1).astype(np.float64, copy=False)
    known_argmax = np.asarray(
        [known_attack_indices[int(i)] for i in known_rel_argmax.tolist()],
        dtype=np.int32,
    )
    attack_threshold = float(config["attack_threshold"])
    known_threshold = float(config["known_attack_confidence_threshold"])
    unknown_index = int(config["unknown_class_index"])

    predicted = np.where(
        scores < attack_threshold,
        int(task_spec.clean_class_index),
        np.where(known_confidence < known_threshold, unknown_index, known_argmax),
    ).astype(np.int32, copy=False)

    return {
        "predicted_labels": predicted,
        "known_attack_confidence": known_confidence,
        "known_attack_argmax": known_argmax,
        "is_unknown_attack": (predicted == unknown_index),
    }


def _open_set_true_labels(
    labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
    config: dict[str, Any] | None,
) -> np.ndarray | None:
    if config is None:
        return None
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    known_attack_indices = set(int(x) for x in config.get("known_attack_indices", []))
    unknown_index = int(config["unknown_class_index"])
    mapped: list[int] = []
    for label in labels_np.tolist():
        label_index = int(label)
        if label_index < 0:
            mapped.append(-1)
        elif label_index == task_spec.clean_class_index or label_index in known_attack_indices:
            mapped.append(label_index)
        elif 0 <= label_index < task_spec.n_classes:
            mapped.append(unknown_index)
        else:
            mapped.append(-1)
    return np.asarray(mapped, dtype=np.int32)


def _summarize_open_set_partition(
    *,
    labels_true: np.ndarray | None,
    open_set_result: dict[str, np.ndarray] | None,
    task_spec: SupervisedTaskSpec,
    config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if open_set_result is None or config is None:
        return None

    predicted = np.asarray(open_set_result["predicted_labels"], dtype=np.int32).reshape(-1)
    labels = [*task_spec.class_names, OPEN_SET_UNKNOWN_ATTACK_NAME]
    label_indices = np.arange(task_spec.n_classes + 1, dtype=np.int32)
    summary: dict[str, Any] = {
        "class_names": [str(x) for x in labels],
        "unknown_class_index": int(config["unknown_class_index"]),
        "predicted_class_distribution": _open_set_distribution_rows(
            predicted,
            task_spec=task_spec,
        ),
        "n_unknown_attack_predictions": int(np.sum(predicted == int(config["unknown_class_index"]))),
        "known_attack_confidence_summary": summarize_scores(
            np.asarray(open_set_result["known_attack_confidence"], dtype=np.float64)
        ),
    }
    if labels_true is None:
        return summary

    truth_all = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    known_mask = truth_all >= 0
    if not bool(np.any(known_mask)):
        return summary
    truth = truth_all[known_mask]
    predicted_known = predicted[known_mask]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted_known,
        labels=label_indices,
        zero_division=0,
    )
    summary["accuracy"] = float(accuracy_score(truth, predicted_known))
    summary["macro_f1"] = float(
        f1_score(truth, predicted_known, labels=label_indices, average="macro", zero_division=0)
    )
    summary["micro_f1"] = float(
        f1_score(truth, predicted_known, labels=label_indices, average="micro", zero_division=0)
    )
    summary["true_class_distribution"] = _open_set_distribution_rows(truth, task_spec=task_spec)
    summary["confusion_matrix"] = {
        "labels": [str(x) for x in labels],
        "rows": confusion_matrix(
            truth,
            predicted_known,
            labels=label_indices,
        ).astype(np.int32, copy=False).tolist(),
    }
    summary["per_class"] = [
        {
            "class_index": int(class_index),
            "class_name": _open_set_class_name(int(class_index), task_spec=task_spec),
            "precision": float(precision[int(pos)]),
            "recall": float(recall[int(pos)]),
            "f1": float(f1[int(pos)]),
            "support": int(support[int(pos)]),
            "predicted_count": int(np.sum(predicted_known == int(class_index))),
        }
        for pos, class_index in enumerate(label_indices.tolist())
    ]
    return summary


def _multiclass_score_extra_rows(
    *,
    task_labels: list[int | None],
    outputs: SupervisedPredictionOutputs,
    task_spec: SupervisedTaskSpec,
    open_set_result: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]] | None:
    if task_spec.is_binary or outputs.probabilities is None:
        return None

    probabilities = np.asarray(outputs.probabilities, dtype=np.float64)
    predicted = np.asarray(outputs.predicted_labels, dtype=np.int32).reshape(-1)
    rows: list[dict[str, Any]] = []
    for row_idx, raw_label in enumerate(task_labels):
        predicted_index = int(predicted[row_idx])
        row: dict[str, Any] = {
            "task_label": "" if raw_label is None else int(raw_label),
            "task_class_name": (
                "" if raw_label is None else _open_set_class_name(int(raw_label), task_spec=task_spec)
            ),
            "predicted_class": predicted_index,
            "predicted_class_name": _open_set_class_name(predicted_index, task_spec=task_spec),
        }
        for class_index, class_name in enumerate(task_spec.class_names):
            row[f"prob_{_csv_safe_class_name(str(class_name))}"] = float(
                probabilities[row_idx, int(class_index)]
            )
        if open_set_result is not None:
            open_pred = int(open_set_result["predicted_labels"][row_idx])
            row.update(
                {
                    "open_set_prediction": open_pred,
                    "open_set_prediction_name": _open_set_class_name(open_pred, task_spec=task_spec),
                    "open_set_is_unknown_attack": bool(open_set_result["is_unknown_attack"][row_idx]),
                    "known_attack_confidence": float(open_set_result["known_attack_confidence"][row_idx]),
                    "known_attack_argmax": int(open_set_result["known_attack_argmax"][row_idx]),
                    "known_attack_argmax_name": _open_set_class_name(
                        int(open_set_result["known_attack_argmax"][row_idx]),
                        task_spec=task_spec,
                    ),
                }
            )
        rows.append(row)
    return rows


def _summarize_multiclass_partition(
    *,
    labels_true: np.ndarray | None,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray | None,
    task_spec: SupervisedTaskSpec,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_labels, dtype=np.int32).reshape(-1)
    summary: dict[str, Any] = {
        "class_names": [str(x) for x in task_spec.class_names],
        "predicted_class_distribution": _distribution_rows_from_predictions(
            predicted,
            task_spec=task_spec,
        ),
    }
    if probabilities is not None:
        summary["mean_probability_by_class"] = [
            {
                "class_index": int(class_index),
                "class_name": str(task_spec.class_names[int(class_index)]),
                "mean_probability": float(np.mean(probabilities[:, class_index])),
            }
            for class_index in range(task_spec.n_classes)
        ]
    if labels_true is None:
        return summary

    truth = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=np.arange(task_spec.n_classes, dtype=np.int32),
        zero_division=0,
    )
    summary["accuracy"] = float(accuracy_score(truth, predicted))
    summary["macro_f1"] = float(f1_score(truth, predicted, average="macro", zero_division=0))
    summary["micro_f1"] = float(f1_score(truth, predicted, average="micro", zero_division=0))
    summary["true_class_distribution"] = _named_label_count_rows(truth, task_spec=task_spec)
    summary["confusion_matrix"] = {
        "labels": [str(x) for x in task_spec.class_names],
        "rows": confusion_matrix(
            truth,
            predicted,
            labels=np.arange(task_spec.n_classes, dtype=np.int32),
        ).astype(np.int32, copy=False).tolist(),
    }
    summary["per_class"] = [
        {
            "class_index": int(class_index),
            "class_name": str(task_spec.class_names[int(class_index)]),
            "precision": float(precision[int(class_index)]),
            "recall": float(recall[int(class_index)]),
            "f1": float(f1[int(class_index)]),
            "support": int(support[int(class_index)]),
            "predicted_count": int(np.sum(predicted == int(class_index))),
        }
        for class_index in range(task_spec.n_classes)
    ]
    return summary


def _build_single_manifest_split_summary(
    *,
    labels: np.ndarray,
    train_indices: np.ndarray,
    infer_indices: np.ndarray,
    train_split_percent: int,
    random_state: int,
    strategy: str,
) -> dict[str, Any]:
    train_labels = labels[train_indices]
    infer_labels = labels[infer_indices] if infer_indices.size > 0 else np.asarray([], dtype=np.int32)
    return {
        "strategy": strategy,
        "requested_train_split_percent": int(train_split_percent),
        "random_state": int(random_state),
        "n_train": int(train_indices.size),
        "n_inference": int(infer_indices.size),
        "train_label_counts": _label_count_rows(np.asarray(train_labels, dtype=np.int32)),
        "inference_label_counts": _label_count_rows(np.asarray(infer_labels, dtype=np.int32)),
    }


def _build_calibration_split_summary(
    *,
    labels: np.ndarray,
    fit_train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
    strategy: str,
) -> dict[str, Any]:
    fit_train_labels = labels[fit_train_indices]
    calibration_labels = labels[calibration_indices]
    return {
        "strategy": strategy,
        "requested_calibration_split_percent": int(calibration_split_percent),
        "random_state": int(random_state),
        "n_fit_train": int(fit_train_indices.size),
        "n_calibration": int(calibration_indices.size),
        "fit_train_label_counts": _label_count_rows(np.asarray(fit_train_labels, dtype=np.int32)),
        "calibration_label_counts": _label_count_rows(np.asarray(calibration_labels, dtype=np.int32)),
    }


def _resolve_holdout_train_count(
    *,
    bucket_size: int,
    train_split_percent: int,
    bucket_name: str,
    split_name: str,
) -> tuple[int, bool]:
    if bucket_size < 2:
        raise ValueError(
            f"{split_name} requires at least two samples in each split bucket when creating a holdout split, "
            f"but '{bucket_name}' has {bucket_size}"
        )

    raw_train = (int(bucket_size) * int(train_split_percent)) / 100.0
    rounded_train = _round_half_up(raw_train)
    train_count = min(max(rounded_train, 1), int(bucket_size) - 1)
    return int(train_count), bool(train_count != rounded_train)


def _build_single_manifest_stratified_split(
    *,
    labels: np.ndarray,
    train_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    if train_split_percent == 100:
        split_summary = _build_single_manifest_split_summary(
            labels=labels,
            train_indices=all_indices,
            infer_indices=np.asarray([], dtype=np.int64),
            train_split_percent=train_split_percent,
            random_state=random_state,
            strategy="single_manifest_all_train",
        )
        split_summary["split_by_folder"] = False
        return (
            all_indices,
            np.asarray([], dtype=np.int64),
            [],
            split_summary,
        )

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Stratified --train-split requires at least two classes in a single manifest")

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    infer_parts: list[np.ndarray] = []
    adjusted_labels: list[int] = []
    per_class_rows: list[dict[str, Any]] = []

    for label, count in zip(unique.tolist(), counts.tolist()):
        label_value = int(label)
        label_count = int(count)
        train_count, adjusted = _resolve_holdout_train_count(
            bucket_size=label_count,
            train_split_percent=train_split_percent,
            bucket_name=f"label {label_value}",
            split_name="Stratified --train-split",
        )
        infer_count = label_count - train_count
        if adjusted:
            adjusted_labels.append(label_value)

        label_indices = all_indices[labels == label_value]
        shuffled = label_indices[rng.permutation(label_indices.shape[0])]
        train_parts.append(shuffled[:train_count])
        infer_parts.append(shuffled[train_count:])
        per_class_rows.append(
            {
                "label": label_value,
                "total": label_count,
                "n_train": int(train_count),
                "n_inference": int(infer_count),
            }
        )

    train_indices = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    infer_indices = np.sort(np.concatenate(infer_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_labels:
        adjusted_preview = ", ".join(str(label) for label in adjusted_labels[:5])
        warnings.append(
            "Adjusted rounded train counts to keep at least one sample in both train and inference "
            f"for labels: {adjusted_preview}"
        )

    split_summary = _build_single_manifest_split_summary(
        labels=labels,
        train_indices=train_indices,
        infer_indices=infer_indices,
        train_split_percent=train_split_percent,
        random_state=random_state,
        strategy="single_manifest_stratified_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["per_class"] = per_class_rows
    return train_indices, infer_indices, warnings, split_summary


def _split_folder_name(item: Any) -> str:
    parent_name = item.model_dir.parent.name.strip()
    if parent_name:
        return parent_name
    return item.model_dir.name


def _build_single_manifest_folder_label_split(
    *,
    items: list[Any],
    labels: np.ndarray,
    train_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if labels.shape[0] != len(items):
        raise ValueError("Folder-based split requires one label per manifest item")

    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    if train_split_percent < 100 and np.unique(labels).size < 2:
        raise ValueError("Folder-based --train-split requires at least two classes in a single manifest")

    bucket_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx, item in enumerate(items):
        folder_name = _split_folder_name(item)
        label_value = int(labels[idx])
        bucket_to_indices.setdefault((folder_name, label_value), []).append(int(idx))

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    infer_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for folder_name, label_value in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(folder_name, label_value)], dtype=np.int64)
        if train_split_percent == 100:
            train_count = int(bucket_indices.size)
            infer_count = 0
            train_parts.append(bucket_indices)
            infer_parts.append(np.asarray([], dtype=np.int64))
        else:
            train_count, adjusted = _resolve_holdout_train_count(
                bucket_size=int(bucket_indices.size),
                train_split_percent=train_split_percent,
                bucket_name=f"{folder_name} [label {label_value}]",
                split_name="Folder-based --train-split",
            )
            infer_count = int(bucket_indices.size) - train_count
            if adjusted:
                adjusted_buckets.append(f"{folder_name} [label {label_value}]")

            shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
            train_parts.append(shuffled[:train_count])
            infer_parts.append(shuffled[train_count:])

        per_bucket_rows.append(
            {
                "folder": folder_name,
                "label": int(label_value),
                "total": int(bucket_indices.size),
                "n_train": int(train_count),
                "n_inference": int(infer_count),
            }
        )

    train_indices = (
        np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
        if train_parts
        else np.asarray([], dtype=np.int64)
    )
    infer_indices = (
        np.sort(np.concatenate(infer_parts).astype(np.int64, copy=False))
        if infer_parts
        else np.asarray([], dtype=np.int64)
    )

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded train counts to keep at least one sample in both train and inference "
            f"for folder/label buckets: {adjusted_preview}"
        )

    strategy = (
        "single_manifest_folder_label_holdout"
        if train_split_percent < 100
        else "single_manifest_folder_label_all_train"
    )
    split_summary = _build_single_manifest_split_summary(
        labels=labels,
        train_indices=train_indices,
        infer_indices=infer_indices,
        train_split_percent=train_split_percent,
        random_state=random_state,
        strategy=strategy,
    )
    split_summary["split_by_folder"] = True
    split_summary["folder_count"] = int(len({row["folder"] for row in per_bucket_rows}))
    split_summary["folder_label_buckets"] = per_bucket_rows
    return train_indices, infer_indices, warnings, split_summary


def _build_calibration_stratified_split(
    *,
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    unique, counts = np.unique(candidate_labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Calibration split requires at least two classes in the training partition")

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_labels: list[int] = []
    per_class_rows: list[dict[str, Any]] = []

    for label, count in zip(unique.tolist(), counts.tolist()):
        label_value = int(label)
        label_indices = candidate_indices[candidate_labels == label_value]
        calibration_count, adjusted = _resolve_holdout_subset_count(
            bucket_size=int(count),
            subset_percent=calibration_split_percent,
            bucket_name=f"label {label_value}",
            split_name="Stratified --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(label_indices.size) - calibration_count
        if adjusted:
            adjusted_labels.append(label_value)

        shuffled = label_indices[rng.permutation(label_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_class_rows.append(
            {
                "label": label_value,
                "total": int(count),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_labels:
        adjusted_preview = ", ".join(str(label) for label in adjusted_labels[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for labels: {adjusted_preview}"
        )

    split_summary = _build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_stratified_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["per_class"] = per_class_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def _rank_label_buckets(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    ranks_np = np.asarray(ranks, dtype=np.int64).reshape(-1)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    return {
        (rank, label): [
            int(i)
            for i in candidate_indices.tolist()
            if int(ranks_np[int(i)]) == int(rank) and int(labels_np[int(i)]) == int(label)
        ]
        for rank in sorted({int(ranks_np[int(i)]) for i in candidate_indices.tolist()})
        for label in sorted(
            {
                int(labels_np[int(i)])
                for i in candidate_indices.tolist()
                if int(ranks_np[int(i)]) == int(rank)
            }
        )
    }


def _rank_label_stratification_keys(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    ranks_np = np.asarray(ranks, dtype=np.int64).reshape(-1)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    return np.asarray(
        [
            f"rank_{int(ranks_np[int(i)])}_label_{int(labels_np[int(i)])}"
            for i in np.asarray(candidate_indices, dtype=np.int64).tolist()
        ],
        dtype=object,
    )


def _rank_label_stratification_warning(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> str | None:
    keys = _rank_label_stratification_keys(
        candidate_indices=candidate_indices,
        ranks=ranks,
        labels=labels,
    )
    unique, counts = np.unique(keys, return_counts=True)
    if unique.size < 2:
        return (
            f"cnn_1d_dann could not use rank-label stratification for {split_name}: "
            "fewer than two rank-label buckets; falling back to label-only stratification"
        )
    min_count = int(np.min(counts))
    if min_count < 2:
        return (
            f"cnn_1d_dann could not use rank-label stratification for {split_name}: "
            f"smallest rank-label bucket has {min_count} row; falling back to label-only stratification"
        )
    return None


def _folder_label_stratification_keys(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    return np.asarray(
        [
            f"folder_{_split_folder_name(items[int(i)])}_label_{int(labels_np[int(i)])}"
            for i in np.asarray(candidate_indices, dtype=np.int64).tolist()
        ],
        dtype=object,
    )


def _folder_label_stratification_warning(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> str | None:
    keys = _folder_label_stratification_keys(
        items=items,
        candidate_indices=candidate_indices,
        labels=labels,
    )
    unique, counts = np.unique(keys, return_counts=True)
    if unique.size < 2:
        return (
            f"Could not use folder-label stratification for {split_name}: "
            "fewer than two folder-label buckets; falling back to label-only stratification"
        )
    min_count = int(np.min(counts))
    if min_count < 2:
        return (
            f"Could not use folder-label stratification for {split_name}: "
            f"smallest folder-label bucket has {min_count} row; falling back to label-only stratification"
        )
    return None


def _build_calibration_rank_label_split(
    *,
    candidate_indices: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    bucket_to_indices = _rank_label_buckets(
        candidate_indices=candidate_indices,
        ranks=ranks,
        labels=labels,
    )
    if len(bucket_to_indices) < 2:
        raise ValueError("Rank-label calibration split requires at least two rank-label buckets")

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for rank, label in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(rank, label)], dtype=np.int64)
        calibration_count, adjusted = _resolve_holdout_subset_count(
            bucket_size=int(bucket_indices.size),
            subset_percent=calibration_split_percent,
            bucket_name=f"rank {int(rank)} [label {int(label)}]",
            split_name="Rank-label --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(bucket_indices.size) - calibration_count
        if adjusted:
            adjusted_buckets.append(f"rank {int(rank)} [label {int(label)}]")

        shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_bucket_rows.append(
            {
                "rank": int(rank),
                "label": int(label),
                "total": int(bucket_indices.size),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for rank-label buckets: {adjusted_preview}"
        )

    split_summary = _build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_rank_label_holdout",
    )
    split_summary["split_by_folder"] = False
    split_summary["split_by_rank_label"] = True
    split_summary["rank_label_buckets"] = per_bucket_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def _build_calibration_folder_label_split(
    *,
    items: list[Any],
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    calibration_split_percent: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if labels.shape[0] != len(items):
        raise ValueError("Folder-based calibration split requires one label per manifest item")

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    if np.unique(candidate_labels).size < 2:
        raise ValueError("Folder-based calibration split requires at least two classes in the training partition")

    bucket_to_indices: dict[tuple[str, int], list[int]] = {}
    for idx in candidate_indices.tolist():
        folder_name = _split_folder_name(items[int(idx)])
        label_value = int(labels[int(idx)])
        bucket_to_indices.setdefault((folder_name, label_value), []).append(int(idx))

    rng = np.random.default_rng(random_state)
    fit_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    adjusted_buckets: list[str] = []
    per_bucket_rows: list[dict[str, Any]] = []

    for folder_name, label_value in sorted(bucket_to_indices):
        bucket_indices = np.asarray(bucket_to_indices[(folder_name, label_value)], dtype=np.int64)
        calibration_count, adjusted = _resolve_holdout_subset_count(
            bucket_size=int(bucket_indices.size),
            subset_percent=calibration_split_percent,
            bucket_name=f"{folder_name} [label {label_value}]",
            split_name="Folder-based --calibration-split",
            subset_label="calibration",
        )
        fit_count = int(bucket_indices.size) - calibration_count
        if adjusted:
            adjusted_buckets.append(f"{folder_name} [label {label_value}]")

        shuffled = bucket_indices[rng.permutation(bucket_indices.shape[0])]
        fit_parts.append(shuffled[:fit_count])
        calibration_parts.append(shuffled[fit_count:])
        per_bucket_rows.append(
            {
                "folder": folder_name,
                "label": int(label_value),
                "total": int(bucket_indices.size),
                "n_fit_train": int(fit_count),
                "n_calibration": int(calibration_count),
            }
        )

    fit_train_indices = np.sort(np.concatenate(fit_parts).astype(np.int64, copy=False))
    calibration_indices = np.sort(np.concatenate(calibration_parts).astype(np.int64, copy=False))

    warnings: list[str] = []
    if adjusted_buckets:
        adjusted_preview = ", ".join(adjusted_buckets[:5])
        warnings.append(
            "Adjusted rounded calibration counts to keep at least one sample in both fit-train and "
            f"calibration for folder/label buckets: {adjusted_preview}"
        )

    split_summary = _build_calibration_split_summary(
        labels=labels,
        fit_train_indices=fit_train_indices,
        calibration_indices=calibration_indices,
        calibration_split_percent=calibration_split_percent,
        random_state=random_state,
        strategy="train_partition_folder_label_holdout",
    )
    split_summary["split_by_folder"] = True
    split_summary["folder_count"] = int(len({row["folder"] for row in per_bucket_rows}))
    split_summary["folder_label_buckets"] = per_bucket_rows
    return fit_train_indices, calibration_indices, warnings, split_summary


def _threshold_candidate_values(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Threshold selection requires at least one score")
    unique_scores = np.unique(scores)
    above_max = np.nextafter(float(np.max(unique_scores)), float("inf"))
    return np.concatenate((np.asarray([above_max], dtype=np.float64), unique_scores[::-1]))


def _evaluate_binary_threshold(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels_true = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_true.shape[0] != scores.shape[0]:
        raise ValueError("Threshold evaluation requires labels and scores with the same length")

    positives = int(np.sum(labels_true == 1))
    negatives = int(np.sum(labels_true == 0))
    flagged = scores >= float(threshold)

    tp = int(np.sum((labels_true == 1) & flagged))
    fp = int(np.sum((labels_true == 0) & flagged))
    tn = int(np.sum((labels_true == 0) & (~flagged)))
    fn = int(np.sum((labels_true == 1) & (~flagged)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / positives) if positives > 0 else 0.0
    false_positive_rate = float(fp / negatives) if negatives > 0 else 0.0
    specificity = float(tn / negatives) if negatives > 0 else 0.0
    accuracy = float((tp + tn) / max(1, labels_true.shape[0]))
    balanced_accuracy = float((recall + specificity) / 2.0)
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_positive": positives,
        "n_negative": negatives,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "fraction_flagged": float(np.mean(flagged)) if flagged.size > 0 else 0.0,
    }


def _select_threshold_max_recall_under_fpr(
    *,
    labels_true: np.ndarray,
    scores: np.ndarray,
    accepted_fpr: float,
) -> dict[str, Any]:
    labels_true = np.asarray(labels_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_true.shape[0] != scores.shape[0]:
        raise ValueError("Threshold selection requires labels and scores with the same length")

    candidates = [
        _evaluate_binary_threshold(labels_true=labels_true, scores=scores, threshold=float(threshold))
        for threshold in _threshold_candidate_values(scores)
    ]
    feasible = [
        row for row in candidates
        if float(row["false_positive_rate"]) <= float(accepted_fpr) + 1e-12
    ]
    if not feasible:
        raise RuntimeError("No feasible threshold satisfied the accepted_fpr constraint")

    best = max(
        feasible,
        key=lambda row: (
            float(row["recall"]),
            -float(row["false_positive_rate"]),
            float(row["precision"]),
            float(row["threshold"]),
        ),
    )
    selected = dict(best)
    selected["selection_method"] = "maximize_recall_subject_to_fpr"
    selected["accepted_fpr"] = float(accepted_fpr)
    return selected


def _build_selected_threshold_summary(
    *,
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": "maximize_recall_subject_to_fpr",
        "source_partition": "calibration",
        "accepted_fprs": [float(selection["accepted_fpr"]) for selection in selections],
        "selections": selections,
    }
    if len(selections) == 1:
        summary.update(selections[0])
    return summary


def _build_selected_threshold_specs(
    selected_threshold_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(selected_threshold_summary, dict):
        return []

    selections = selected_threshold_summary.get("selections", [])
    if not isinstance(selections, list):
        return []

    specs: list[dict[str, Any]] = []
    selection_method = selected_threshold_summary.get("method")
    source_partition = selected_threshold_summary.get("source_partition")
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        if "threshold" not in selection or "accepted_fpr" not in selection:
            continue
        spec: dict[str, Any] = {
            "accepted_fpr": float(selection["accepted_fpr"]),
            "threshold": float(selection["threshold"]),
        }
        if selection_method is not None:
            spec["selection_method"] = str(selection_method)
        if source_partition is not None:
            spec["source_partition"] = str(source_partition)
        specs.append(spec)
    return specs


def _unique_index_by_name(names: list[str], *, context: str) -> dict[str, int]:
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = int(i)
    if duplicates:
        dup_preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"Duplicate model names in {context}; cannot align features safely. "
            f"Examples: {dup_preview}"
        )
    return index


def _feature_block_name(feature_name: str) -> str:
    block_name, sep, _ = str(feature_name).rpartition(".")
    if not sep or not block_name:
        raise ValueError(f"Invalid spectral feature name in external metadata: {feature_name}")
    return block_name


def _ordered_block_names_from_feature_names(feature_names: list[str]) -> list[str]:
    block_names: list[str] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        block_name = _feature_block_name(feature_name)
        if block_name in seen:
            continue
        seen.add(block_name)
        block_names.append(block_name)
    return block_names


def _build_requested_feature_names(
    *,
    block_names: list[str],
    spectral_features: list[str],
    sv_top_k: int,
    spectral_moment_source: str,
) -> list[str]:
    return build_spectral_feature_names(
        block_names=block_names,
        selected_features=spectral_features,
        sv_top_k=sv_top_k,
        spectral_moment_source=spectral_moment_source,
        shorten_block_names=False,
    )


def _normalized_representation_kind(metadata: dict[str, Any] | None, *, feature_ndim: int | None = None) -> str:
    if isinstance(metadata, dict):
        raw_kind = metadata.get("representation_kind")
        if isinstance(raw_kind, str) and raw_kind.strip():
            return str(raw_kind).strip()
    if feature_ndim == 4:
        return ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND
    return TABULAR_SPECTRAL_REPRESENTATION_KIND


def _slice_supervised_features(features: np.ndarray | SupervisedFeatureBundle, indices: np.ndarray) -> Any:
    resolved_indices = np.asarray(indices, dtype=np.int64)
    if isinstance(features, SupervisedFeatureBundle):
        return features.subset(resolved_indices)
    return np.asarray(features[resolved_indices], dtype=np.float32)


def _is_dann_model_name(model_name: str) -> bool:
    return str(model_name) == CNN_1D_DANN_MODEL_NAME


def _parse_rank_from_model_name(model_name: str) -> int:
    match = _RANK_TOKEN_RE.search(str(model_name))
    if match is None:
        raise ValueError(f"Could not parse a rank token from model name: {model_name!r}")
    return int(float(match.group(1)))


def _domain_class_name_for_rank(rank: int) -> str:
    return f"rank_{int(rank)}"


def _build_dann_domain_adaptation_config(
    *,
    model_names: list[str],
    all_items: list[Any],
    train_indices: np.ndarray,
    infer_indices: np.ndarray,
    labels_values: np.ndarray,
    labels_known: np.ndarray,
    mode: str,
    source_rank: int,
    target_adaptation_percent: int,
    lambda_max: float,
    lambda_gamma: float,
    lr_alpha: float,
    lr_beta: float,
    random_state: int,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if CNN_1D_DANN_MODEL_NAME not in model_names:
        return None, []
    if str(mode) != "joint":
        raise ValueError(
            "cnn_1d_dann requires a joint manifest whose train split contains the seen ranks "
            "and whose infer split contains disjoint zero-shot ranks"
        )

    ranks = np.asarray([_parse_rank_from_model_name(item.model_name) for item in all_items], dtype=np.int64)
    train_indices_np = np.asarray(train_indices, dtype=np.int64)
    infer_indices_np = np.asarray(infer_indices, dtype=np.int64)
    if train_indices_np.size == 0:
        raise ValueError("cnn_1d_dann requires at least one manifest train row")
    if infer_indices_np.size == 0:
        raise ValueError("cnn_1d_dann requires at least one zero-shot inference row")

    labels_known_np = np.asarray(labels_known, dtype=bool).reshape(-1)
    if not bool(np.all(labels_known_np[train_indices_np])):
        raise ValueError("cnn_1d_dann requires known labels for every manifest train row")

    observed_train_ranks = sorted({int(ranks[int(i)]) for i in train_indices_np.tolist()})
    if int(source_rank) not in observed_train_ranks:
        raise ValueError(
            "cnn_1d_dann requires the configured source rank in the manifest train split; "
            f"{int(source_rank)}; observed train ranks={observed_train_ranks}"
        )
    target_train_ranks = [rank for rank in observed_train_ranks if rank != int(source_rank)]
    if not target_train_ranks:
        raise ValueError(
            "cnn_1d_dann requires at least one non-source rank in the manifest train split"
        )
    train_ranks = [int(source_rank), *target_train_ranks]

    zero_shot_inference_ranks = sorted({int(ranks[int(i)]) for i in infer_indices_np.tolist()})
    if not zero_shot_inference_ranks:
        raise ValueError("cnn_1d_dann requires at least one zero-shot inference rank")
    overlapping_ranks = sorted(set(train_ranks) & set(zero_shot_inference_ranks))
    if overlapping_ranks:
        raise ValueError(
            "cnn_1d_dann requires manifest infer ranks to be disjoint from manifest train ranks; "
            f"overlap={overlapping_ranks}"
        )

    domain_rank_values = [int(source_rank), *target_train_ranks]
    rank_to_domain_index = {int(rank): int(idx) for idx, rank in enumerate(domain_rank_values)}
    domain_labels = np.full(int(ranks.shape[0]), -1, dtype=np.int64)
    for idx in train_indices_np.tolist():
        domain_labels[int(idx)] = int(rank_to_domain_index[int(ranks[int(idx)])])
    domain_labels_path = output_dir / "supervised_domain_rank_labels.npy"
    np.save(domain_labels_path, domain_labels)

    labels_np = np.asarray(labels_values, dtype=np.int32).reshape(-1)
    source_train_indices = np.asarray(
        [int(i) for i in train_indices_np.tolist() if int(ranks[int(i)]) == int(source_rank)],
        dtype=np.int64,
    )
    target_train_indices = np.asarray(
        [int(i) for i in train_indices_np.tolist() if int(ranks[int(i)]) != int(source_rank)],
        dtype=np.int64,
    )

    config = {
        "enabled": True,
        "model_name": CNN_1D_DANN_MODEL_NAME,
        "source_rank": int(source_rank),
        "train_ranks": [int(x) for x in train_ranks],
        "target_train_ranks": [int(x) for x in target_train_ranks],
        "zero_shot_inference_ranks": [int(x) for x in zero_shot_inference_ranks],
        "domain_rank_values": [int(x) for x in domain_rank_values],
        "domain_class_names": [_domain_class_name_for_rank(rank) for rank in domain_rank_values],
        "rank_to_domain_index": {str(rank): int(idx) for rank, idx in rank_to_domain_index.items()},
        "source_domain_index": int(rank_to_domain_index[int(source_rank)]),
        "train_indices": [int(x) for x in train_indices_np.tolist()],
        "source_train_indices": [int(x) for x in source_train_indices.tolist()],
        "target_train_indices": [int(x) for x in target_train_indices.tolist()],
        "inference_indices": [int(x) for x in infer_indices_np.tolist()],
        "train_rank_label_counts": [
            {
                "rank": int(rank),
                "label": int(label),
                "count": int(np.sum((ranks[train_indices_np] == int(rank)) & (labels_np[train_indices_np] == int(label)))),
            }
            for rank in train_ranks
            for label in sorted({int(labels_np[int(i)]) for i in train_indices_np.tolist() if int(ranks[int(i)]) == int(rank)})
        ],
        "domain_labels_path": str(domain_labels_path),
        "label_loss_scope": "all_training_ranks",
        "domain_loss": "multiclass_rank_cross_entropy",
        "domain_loss_weight": 1.0,
        "lambda_schedule": {
            "type": "dann_paper_logistic",
            "lambda_max": float(lambda_max),
            "gamma": float(lambda_gamma),
        },
        "learning_rate_schedule": {
            "type": "fixed",
        },
    }
    warnings = [
        "cnn_1d_dann is using manifest-defined rank-adversarial supervised training: every train "
        "row contributes label loss and rank-domain loss; manifest infer rows are zero-shot ranks "
        "excluded from training, normalization, calibration, and threshold fitting"
    ]
    if int(target_adaptation_percent) != int(DANN_DEFAULT_TARGET_ADAPTATION_PERCENT):
        warnings.append(
            "Ignored deprecated --dann-target-adaptation-percent because cnn_1d_dann now derives "
            "seen and zero-shot ranks from the joint manifest train/infer partitions"
        )
    if float(lr_alpha) != float(DANN_DEFAULT_LR_ALPHA) or float(lr_beta) != float(DANN_DEFAULT_LR_BETA):
        warnings.append(
            "Ignored deprecated --dann-lr-alpha/--dann-lr-beta because cnn_1d_dann now uses "
            "a fixed learning rate"
        )
    return config, warnings


def _resolve_dann_fit_inputs(
    *,
    manifest: dict[str, Any],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    domain_cfg = manifest.get("domain_adaptation")
    if not isinstance(domain_cfg, dict) or not bool(domain_cfg.get("enabled")):
        raise ValueError("cnn_1d_dann task is missing prepared domain_adaptation metadata")

    labels_path = domain_cfg.get("domain_labels_path")
    if not isinstance(labels_path, str) or not labels_path:
        raise ValueError("cnn_1d_dann domain_adaptation metadata is missing domain_labels_path")
    domain_labels_all = np.load(labels_path).astype(np.int64)
    fit_indices = np.asarray(train_indices, dtype=np.int64)
    if fit_indices.size == 0:
        raise ValueError("cnn_1d_dann resolved zero training rows for fitting")
    if int(np.max(fit_indices)) >= int(domain_labels_all.shape[0]):
        raise ValueError("cnn_1d_dann training indices exceed the domain-label array length")
    if bool(np.any(domain_labels_all[fit_indices] < 0)):
        raise ValueError(
            "cnn_1d_dann resolved fit rows without training-rank domain labels; "
            "check that only manifest train rows are used for fitting"
        )

    fit_kwargs = {
        "domain_labels": domain_labels_all[fit_indices],
        "domain_class_names": [str(x) for x in domain_cfg.get("domain_class_names", [])],
        "domain_rank_values": [int(x) for x in domain_cfg.get("domain_rank_values", [])],
    }
    return fit_indices, fit_kwargs


def _resolve_rank_label_fit_kwargs(
    *,
    params: dict[str, Any],
    rank_values: np.ndarray | None,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    if not bool(params.get("rank_label_weight_loss", False)):
        return {}
    if rank_values is None:
        raise ValueError("rank_label_weight_loss requires prepared rank values")
    ranks_np = np.asarray(rank_values, dtype=np.int64).reshape(-1)
    fit_indices_np = np.asarray(fit_indices, dtype=np.int64)
    if fit_indices_np.size == 0:
        raise ValueError("rank_label_weight_loss resolved zero training rows for fitting")
    if int(np.max(fit_indices_np)) >= int(ranks_np.shape[0]):
        raise ValueError("rank_label_weight_loss training indices exceed the rank-value array length")
    return {"rank_labels": ranks_np[fit_indices_np]}


def _tabular_array_or_raise(features: np.ndarray | SupervisedFeatureBundle) -> np.ndarray:
    if isinstance(features, SupervisedFeatureBundle):
        raise ValueError("This operation requires a tabular feature matrix, but the run uses a structured bundle")
    return np.asarray(features, dtype=np.float32)


def _supervised_feature_row_count(features: np.ndarray | SupervisedFeatureBundle) -> int:
    if isinstance(features, SupervisedFeatureBundle):
        return int(features.n_samples)
    return int(features.shape[0])


def _compatible_model_names_for_representation(
    *,
    requested_model_name: str,
    representation_kind: str,
) -> tuple[list[str], list[str]]:
    registered = registered_models()
    compatible = [
        name
        for name in registered
        if str(representation_kind) in supported_representation_kinds(name)
        and name != CNN_1D_DANN_MODEL_NAME
    ]
    incompatible = [name for name in registered if name not in compatible]
    if requested_model_name == "all":
        if not compatible:
            raise ValueError(
                "No registered supervised models support representation_kind="
                f"{representation_kind!r}"
            )
        return compatible, incompatible

    if requested_model_name not in registered:
        raise ValueError(
            f"Unknown supervised model '{requested_model_name}'. Registered: {registered}"
        )
    if str(representation_kind) not in supported_representation_kinds(requested_model_name):
        raise ValueError(
            f"Supervised model '{requested_model_name}' does not support representation_kind="
            f"{representation_kind!r}"
        )
    return [requested_model_name], incompatible


def _validate_layer_sequence_external_bundle(
    *,
    metadata: dict[str, Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
) -> dict[str, Any]:
    selected_features = resolve_spectral_features(spectral_features)
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    resolved_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)

    metadata_features = metadata.get("resolved_features")
    if not isinstance(metadata_features, list) or [str(x) for x in metadata_features] != list(selected_features):
        raise ValueError(
            "Structured external feature bundle was already aggregated with a different --features selection"
        )
    if "sv_topk" in selected_features and int(metadata.get("sv_top_k", spectral_sv_top_k)) != int(
        spectral_sv_top_k
    ):
        raise ValueError(
            "Structured external feature bundle was aggregated with a different --spectral-sv-top-k"
        )
    if str(metadata.get("spectral_moment_source", resolved_moment_source)) != str(resolved_moment_source):
        raise ValueError(
            "Structured external feature bundle was aggregated with a different --spectral-moment-source"
        )
    if str(metadata.get("spectral_qv_sum_mode", resolved_qv_sum_mode)) != str(resolved_qv_sum_mode):
        raise ValueError(
            "Structured external feature bundle was aggregated with a different --spectral-qv-sum-mode"
        )
    return sanitize_spectral_metadata(dict(metadata))


def _filter_external_spectral_columns(
    *,
    features: np.ndarray,
    metadata: dict[str, Any],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    raw_feature_names = metadata.get("feature_names")
    if not isinstance(raw_feature_names, list) or not raw_feature_names:
        raise ValueError(
            "External spectral metadata must include non-empty 'feature_names' to honor "
            "--features/--spectral-sv-top-k/--spectral-qv-sum-mode"
        )

    feature_names = [str(x) for x in raw_feature_names]
    if len(feature_names) != int(features.shape[1]):
        raise ValueError(
            f"External feature metadata column count ({len(feature_names)}) does not match "
            f"feature matrix width ({features.shape[1]})"
        )

    selected_features = resolve_spectral_features(spectral_features)
    resolved_moment_source = resolve_spectral_moment_source(spectral_moment_source)
    resolved_qv_sum_mode = resolve_spectral_qv_sum_mode(spectral_qv_sum_mode)
    all_block_names = _ordered_block_names_from_feature_names(feature_names)

    if resolved_qv_sum_mode == "none":
        selected_block_names = [name for name in all_block_names if ".qv_sum" not in name]
    elif resolved_qv_sum_mode == "only":
        selected_block_names = [name for name in all_block_names if ".qv_sum" in name]
    else:
        selected_block_names = list(all_block_names)

    if not selected_block_names:
        raise ValueError(
            "External feature matrix does not contain any blocks compatible with "
            f"--spectral-qv-sum-mode={resolved_qv_sum_mode}"
        )

    expected_feature_names = _build_requested_feature_names(
        block_names=selected_block_names,
        spectral_features=selected_features,
        sv_top_k=int(spectral_sv_top_k),
        spectral_moment_source=resolved_moment_source,
    )
    if not expected_feature_names:
        raise ValueError("Requested external spectral feature selection resolved to zero columns")

    feature_index = _unique_index_by_name(feature_names, context="external spectral feature names")
    missing = [name for name in expected_feature_names if name not in feature_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "External feature matrix does not contain the requested spectral columns. "
            f"Examples: {preview}"
        )

    warnings: list[str] = []
    if feature_names != expected_feature_names:
        column_indices = np.asarray([feature_index[name] for name in expected_feature_names], dtype=np.int64)
        features = features[:, column_indices]
        warnings.append(
            "Filtered/reordered external feature columns to match requested spectral configuration"
        )

    filtered_metadata = dict(metadata)
    filtered_metadata["resolved_features"] = list(selected_features)
    filtered_metadata["spectral_moment_source"] = resolved_moment_source
    filtered_metadata["spectral_qv_sum_mode"] = resolved_qv_sum_mode
    filtered_metadata["sv_top_k"] = int(spectral_sv_top_k)
    filtered_metadata["feature_dim"] = int(features.shape[1])
    filtered_metadata["feature_names"] = list(expected_feature_names)
    filtered_metadata["block_names"] = list(selected_block_names)
    filtered_metadata["n_blocks"] = int(len(selected_block_names))
    source_lora_dims = spectral_block_lora_dims_by_block(metadata)
    if all(block_name in source_lora_dims for block_name in selected_block_names):
        filtered_metadata["lora_adapter_dims"] = [
            source_lora_dims[block_name] for block_name in selected_block_names
        ]

    filtered_metadata["base_block_names"] = [
        name for name in selected_block_names if ".qv_sum" not in name
    ]
    filtered_metadata["qv_sum_block_names"] = [
        name for name in selected_block_names if ".qv_sum" in name
    ]

    extractor_params = filtered_metadata.get("extractor_params")
    if isinstance(extractor_params, dict):
        filtered_params = dict(extractor_params)
        filtered_params["spectral_features"] = list(selected_features)
        filtered_params["spectral_sv_top_k"] = int(spectral_sv_top_k)
        filtered_params["spectral_moment_source"] = resolved_moment_source
        filtered_params["spectral_qv_sum_mode"] = resolved_qv_sum_mode
        filtered_metadata["extractor_params"] = spectral_extractor_params(filtered_params)

    return features, sanitize_spectral_metadata(filtered_metadata), warnings


def _load_external_spectral_bundle(
    *,
    feature_file: Path,
    model_names_file: Path,
    metadata_file: Path | None,
    group_mask_file: Path | None,
    value_mask_file: Path | None,
    group_names_file: Path | None,
    expected_model_names: list[str],
    spectral_features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    allow_manifest_subset: bool = False,
) -> tuple[np.ndarray | SupervisedFeatureBundle, dict[str, Any], list[str], np.ndarray]:
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if not model_names_file.exists():
        raise FileNotFoundError(f"Model names file not found: {model_names_file}")
    if metadata_file is not None and not metadata_file.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {metadata_file}")

    features_mmap = np.load(feature_file, mmap_mode="r")
    if features_mmap.ndim not in {2, 4}:
        raise ValueError(
            f"Expected a 2D or 4D feature bundle at {feature_file}, got shape={features_mmap.shape}"
        )

    with open(model_names_file, "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    if len(model_names) != int(features_mmap.shape[0]):
        raise ValueError(
            f"Model names length ({len(model_names)}) does not match feature rows ({features_mmap.shape[0]}) "
            f"for external features"
        )

    metadata: dict[str, Any] = {}
    if metadata_file is not None:
        loaded = load_spectral_metadata(metadata_file)
        if isinstance(loaded, dict):
            metadata = dict(loaded)
    representation_kind = _normalized_representation_kind(metadata, feature_ndim=int(features_mmap.ndim))
    if str(representation_kind) == ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND and features_mmap.ndim != 4:
        raise ValueError(
            "Structured layer-sequence metadata requires a 4D feature tensor, "
            f"but {feature_file} has shape={features_mmap.shape}"
        )
    if str(representation_kind) != ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND and features_mmap.ndim != 2:
        raise ValueError(
            "Tabular supervised representations require a 2D feature matrix, "
            f"but {feature_file} has shape={features_mmap.shape}"
        )

    ext_index = _unique_index_by_name(model_names, context=str(model_names_file))
    expected_index = _unique_index_by_name(expected_model_names, context="manifest model names")

    missing = sorted(name for name in expected_index if name not in ext_index)
    extra = sorted(name for name in ext_index if name not in expected_index)
    if missing and not allow_manifest_subset:
        details: list[str] = [f"missing={missing[:5]}"]
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "External feature/model-name set does not cover the manifest model-name set: "
            + "; ".join(details)
        )

    selected_expected_indices = np.asarray(
        [i for i, name in enumerate(expected_model_names) if name in ext_index],
        dtype=np.int64,
    )
    if selected_expected_indices.size == 0:
        details: list[str] = []
        if expected_model_names:
            details.append(f"expected={expected_model_names[:5]}")
        if model_names:
            details.append(f"external={model_names[:5]}")
        raise ValueError(
            "External feature/model-name set has no overlap with the manifest model-name set"
            + (": " + "; ".join(details) if details else "")
        )

    selected_expected_model_names = [expected_model_names[int(i)] for i in selected_expected_indices.tolist()]
    row_indices = np.asarray([ext_index[name] for name in selected_expected_model_names], dtype=np.int64)
    features = np.asarray(features_mmap[row_indices], dtype=np.float32)

    requested_rows = int(len(expected_model_names))
    expected_rows = int(len(selected_expected_model_names))
    source_rows = int(features_mmap.shape[0])

    warnings: list[str] = []
    if allow_manifest_subset and missing:
        warning = (
            "Selected external feature rows by manifest model names from a source bundle "
            f"with {source_rows} rows; retained {expected_rows}/{requested_rows} manifest models "
            f"and skipped {len(missing)} missing model names"
        )
        if extra:
            warning += f"; ignored {len(extra)} external-only model names"
        warning += f". Missing example(s): {missing[:5]}"
        warnings.append(warning)
    elif source_rows != expected_rows:
        warnings.append(
            "Selected external feature rows by manifest model names from a source bundle "
            f"with {source_rows} rows"
        )
    elif selected_expected_model_names != model_names:
        warnings.append("Reordered external features to match manifest model order using model names")

    metadata = dict(metadata)
    metadata["representation_kind"] = str(representation_kind)
    metadata["n_models"] = int(features.shape[0])
    metadata["external_manifest_requested_n_models"] = int(requested_rows)
    metadata["external_manifest_selected_n_models"] = int(features.shape[0])
    metadata["external_manifest_missing_model_count"] = int(len(missing))
    if source_rows != int(features.shape[0]):
        metadata["external_source_n_models"] = int(source_rows)

    if str(representation_kind) == ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND:
        if group_mask_file is None or not group_mask_file.exists():
            raise FileNotFoundError(
                "Structured external feature bundle is missing the required group-mask companion"
            )
        if value_mask_file is None or not value_mask_file.exists():
            raise FileNotFoundError(
                "Structured external feature bundle is missing the required value-mask companion"
            )
        if group_names_file is None or not group_names_file.exists():
            raise FileNotFoundError(
                "Structured external feature bundle is missing the required group-names companion"
            )

        group_mask_mmap = np.load(group_mask_file, mmap_mode="r")
        value_mask_mmap = np.load(value_mask_file, mmap_mode="r")
        if group_mask_mmap.shape != features_mmap.shape[:2]:
            raise ValueError(
                f"group_mask shape {group_mask_mmap.shape} does not match feature tensor shape {features_mmap.shape[:2]}"
            )
        if value_mask_mmap.shape != features_mmap.shape:
            raise ValueError(
                f"value_mask shape {value_mask_mmap.shape} does not match feature tensor shape {features_mmap.shape}"
            )
        with open(group_names_file, "r", encoding="utf-8") as f:
            raw_group_names = json.load(f)
        if not isinstance(raw_group_names, list) or len(raw_group_names) != source_rows:
            raise ValueError(
                "Structured external feature bundle group-names companion must be a row-aligned JSON list"
            )

        selected_group_names = [
            [str(x) for x in raw_group_names[int(source_idx)]]
            for source_idx in row_indices.tolist()
        ]
        metadata = _validate_layer_sequence_external_bundle(
            metadata=metadata,
            spectral_features=spectral_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
        )
        metadata["structural_group_names"] = [list(names) for names in selected_group_names]
        return (
            SupervisedFeatureBundle(
                values=features,
                representation_kind=str(representation_kind),
                metadata=dict(metadata),
                group_mask=np.asarray(group_mask_mmap[row_indices], dtype=bool),
                value_mask=np.asarray(value_mask_mmap[row_indices], dtype=bool),
                group_names=selected_group_names,
            ),
            metadata,
            warnings,
            selected_expected_indices,
        )

    features, metadata, column_warnings = _filter_external_spectral_columns(
        features=features,
        metadata=metadata,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
    )
    warnings.extend(column_warnings)

    return features, metadata, warnings, selected_expected_indices


def _load_features_for_tuning_manifest(
    manifest: dict[str, Any],
) -> np.ndarray | SupervisedFeatureBundle:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest is missing data configuration")

    feature_loading_mode = str(data.get("feature_loading_mode", "materialized"))
    feature_path_value = data.get("feature_path")
    if not isinstance(feature_path_value, str) or not feature_path_value:
        raise ValueError("Tuning manifest is missing data.feature_path")

    if feature_loading_mode == "materialized":
        return np.asarray(np.load(feature_path_value), dtype=np.float32)

    if feature_loading_mode != "external_source":
        raise ValueError(f"Unsupported data.feature_loading_mode={feature_loading_mode!r}")

    model_names_path_value = data.get("model_names_path")
    if not isinstance(model_names_path_value, str) or not model_names_path_value:
        raise ValueError("Tuning manifest is missing data.model_names_path")

    with open(model_names_path_value, "r", encoding="utf-8") as f:
        expected_model_names = [str(x) for x in json.load(f)]

    extractor = manifest.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("Tuning manifest is missing extractor configuration")

    extractor_params = extractor.get("params")
    if not isinstance(extractor_params, dict):
        raise ValueError("Tuning manifest is missing extractor.params")

    extractor_metadata = extractor.get("metadata")
    if not isinstance(extractor_metadata, dict):
        raise ValueError("Tuning manifest is missing extractor.metadata")

    external_feature_source = extractor_metadata.get("external_feature_source")
    external_model_names_source = extractor_metadata.get("external_model_names_source")
    external_metadata_source = extractor_metadata.get("external_metadata_source")
    external_group_mask_source = extractor_metadata.get("external_group_mask_source")
    external_value_mask_source = extractor_metadata.get("external_value_mask_source")
    external_group_names_source = extractor_metadata.get("external_group_names_source")
    if not isinstance(external_feature_source, str) or not external_feature_source:
        raise ValueError("Tuning manifest is missing extractor.metadata.external_feature_source")
    if not isinstance(external_model_names_source, str) or not external_model_names_source:
        raise ValueError("Tuning manifest is missing extractor.metadata.external_model_names_source")

    spectral_features_value = extractor_params.get("spectral_features")
    if isinstance(spectral_features_value, list) and spectral_features_value:
        spectral_features = [str(x) for x in spectral_features_value]
    else:
        spectral_features = resolve_spectral_features(None)

    features, _, _, selected_expected_indices = _load_external_spectral_bundle(
        feature_file=Path(external_feature_source),
        model_names_file=Path(external_model_names_source),
        metadata_file=(
            Path(external_metadata_source)
            if isinstance(external_metadata_source, str) and external_metadata_source
            else None
        ),
        group_mask_file=(
            Path(external_group_mask_source)
            if isinstance(external_group_mask_source, str) and external_group_mask_source
            else None
        ),
        value_mask_file=(
            Path(external_value_mask_source)
            if isinstance(external_value_mask_source, str) and external_value_mask_source
            else None
        ),
        group_names_file=(
            Path(external_group_names_source)
            if isinstance(external_group_names_source, str) and external_group_names_source
            else None
        ),
        expected_model_names=expected_model_names,
        spectral_features=spectral_features,
        spectral_sv_top_k=int(extractor_params.get("spectral_sv_top_k", 8)),
        spectral_moment_source=str(
            extractor_params.get("spectral_moment_source", DEFAULT_SPECTRAL_MOMENT_SOURCE)
        ),
        spectral_qv_sum_mode=str(
            extractor_params.get("spectral_qv_sum_mode", DEFAULT_SPECTRAL_QV_SUM_MODE)
        ),
    )
    if int(selected_expected_indices.size) != int(len(expected_model_names)):
        raise ValueError(
            "External feature source no longer covers the prepared tuning-manifest model-name set; "
            "regenerate the supervised prepare stage to refresh the subset of available models"
        )
    return features


def _compact_indices_to_selected_scope(
    indices: np.ndarray,
    *,
    selected_expected_indices: np.ndarray,
) -> np.ndarray:
    remap = {int(old): int(new) for new, old in enumerate(selected_expected_indices.tolist())}
    compacted = [remap[int(idx)] for idx in indices.tolist() if int(idx) in remap]
    return np.asarray(compacted, dtype=np.int64)


def _validated_train_label_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("Supervised classification requires at least two classes in the training set")
    return unique, counts


def _sanitize_cv_folds(labels: np.ndarray, requested_folds: int) -> tuple[int, list[str]]:
    if requested_folds < 2:
        raise ValueError(f"cv_folds must be >=2, got {requested_folds}")

    _unique, counts = _validated_train_label_counts(labels)
    min_count = int(np.min(counts))
    warnings: list[str] = []
    if min_count < 2:
        warnings.append(
            "Minority class count <2; falling back to a single train-as-validation split for tuning"
        )
        return 1, warnings

    resolved = min(requested_folds, min_count)
    if resolved != requested_folds:
        warnings.append(
            f"Reduced cv_folds from {requested_folds} to {resolved} due to minority class size={min_count}"
        )
    return int(resolved), warnings


def _build_cv_splits(
    *,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if cv_folds == 1:
        return [
            {
                "split_index": 0,
                "train_indices": [int(x) for x in train_indices.tolist()],
                "valid_indices": [int(x) for x in train_indices.tolist()],
            }
        ]

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits: list[dict[str, Any]] = []
    dummy = np.zeros((train_indices.shape[0], 1), dtype=np.float32)
    for split_idx, (tr_rel, val_rel) in enumerate(splitter.split(dummy, train_labels)):
        tr_abs = train_indices[np.asarray(tr_rel, dtype=np.int64)]
        val_abs = train_indices[np.asarray(val_rel, dtype=np.int64)]
        splits.append(
            {
                "split_index": int(split_idx),
                "train_indices": [int(x) for x in tr_abs.tolist()],
                "valid_indices": [int(x) for x in val_abs.tolist()],
            }
        )
    return splits


def _estimate_total_fit_count(
    *,
    n_tasks: int,
    cv_split_groups: list[dict[str, Any]],
) -> int:
    total_splits = sum(len(group.get("cv_splits", [])) for group in cv_split_groups)
    return int(n_tasks * total_splits)


def _grid_search_warnings(
    *,
    n_tasks: int,
    estimated_total_fits: int,
) -> list[str]:
    warnings: list[str] = []
    if n_tasks > 100:
        warnings.append(
            "Large supervised grid search: "
            f"{n_tasks} tasks and approximately {estimated_total_fits} total model fits across CV splits"
    )
    return warnings


def _unwrap_final_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps") and getattr(model, "named_steps", None):
        return list(model.named_steps.values())[-1]
    if hasattr(model, "steps") and getattr(model, "steps", None):
        return model.steps[-1][1]
    return model


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def _softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    normalizer = np.sum(exp_logits, axis=1, keepdims=True)
    normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
    return np.asarray(exp_logits / normalizer, dtype=np.float64)


def _observed_model_classes(model: Any) -> np.ndarray | None:
    estimator = _unwrap_final_estimator(model)
    for candidate in (estimator, model):
        classes = getattr(candidate, "classes_", None)
        if classes is not None:
            return np.asarray(classes, dtype=np.int32).reshape(-1)
    return None


def _align_task_matrix(
    values: np.ndarray,
    *,
    observed_classes: np.ndarray | None,
    task_spec: SupervisedTaskSpec,
    fill_value: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D class-output matrix, got shape={matrix.shape}")
    if matrix.shape[1] == task_spec.n_classes and observed_classes is None:
        return np.asarray(matrix, dtype=np.float64)
    if observed_classes is None:
        observed_classes = np.arange(matrix.shape[1], dtype=np.int32)
    if int(observed_classes.shape[0]) != int(matrix.shape[1]):
        raise ValueError(
            "Model output columns do not align with model classes_: "
            f"shape={matrix.shape}, classes={observed_classes.tolist()}"
        )
    aligned = np.full((matrix.shape[0], task_spec.n_classes), float(fill_value), dtype=np.float64)
    for source_col, class_value in enumerate(observed_classes.tolist()):
        class_index = int(class_value)
        if class_index < 0 or class_index >= task_spec.n_classes:
            raise ValueError(
                f"Observed class index {class_index} is outside the configured task classes "
                f"[0, {task_spec.n_classes - 1}]"
            )
        aligned[:, class_index] = matrix[:, source_col]
    return aligned


def _predict_scores(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "predict_scores"):
        scores = np.asarray(model.predict_scores(x), dtype=np.float64)
        return scores.reshape(-1)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=np.float64)
        if decision.ndim == 2 and decision.shape[1] >= 2:
            return decision[:, 1]
        return decision.reshape(-1)

    pred = np.asarray(model.predict(x), dtype=np.float64)
    return pred.reshape(-1)


def _predict_task_outputs(
    model: Any,
    x: Any,
    *,
    task_spec: SupervisedTaskSpec,
) -> SupervisedPredictionOutputs:
    if task_spec.is_binary:
        probabilities: np.ndarray | None = None
        logits: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            observed_classes = _observed_model_classes(model)
            raw_proba = np.asarray(model.predict_proba(x), dtype=np.float64)
            if raw_proba.ndim == 1:
                raw_proba = raw_proba.reshape(-1, 1)
            probabilities = _align_task_matrix(
                raw_proba,
                observed_classes=observed_classes,
                task_spec=task_spec,
                fill_value=0.0,
            )
            backdoor_scores = probabilities[:, 1]
            clipped = np.clip(backdoor_scores, 1e-12, 1.0 - 1e-12)
            logits = np.log(clipped / (1.0 - clipped))
        elif hasattr(model, "decision_function"):
            raw_decision = np.asarray(model.decision_function(x), dtype=np.float64)
            if raw_decision.ndim == 2 and raw_decision.shape[1] >= 2:
                observed_classes = _observed_model_classes(model)
                aligned = _align_task_matrix(
                    raw_decision,
                    observed_classes=observed_classes,
                    task_spec=task_spec,
                    fill_value=-math.inf,
                )
                backdoor_scores = aligned[:, 1]
                logits = np.asarray(backdoor_scores, dtype=np.float64)
            else:
                backdoor_scores = raw_decision.reshape(-1)
                logits = np.asarray(backdoor_scores, dtype=np.float64)
            positive_prob = _sigmoid(backdoor_scores)
            probabilities = np.column_stack([1.0 - positive_prob, positive_prob]).astype(
                np.float64,
                copy=False,
            )
        else:
            predicted = np.asarray(model.predict(x), dtype=np.int32).reshape(-1)
            backdoor_scores = predicted.astype(np.float64, copy=False)
            probabilities = np.column_stack([1.0 - backdoor_scores, backdoor_scores]).astype(
                np.float64,
                copy=False,
            )
            logits = np.asarray(backdoor_scores, dtype=np.float64)

        if hasattr(model, "predict"):
            predicted_labels = np.asarray(model.predict(x), dtype=np.int32).reshape(-1)
        elif probabilities is not None:
            predicted_labels = np.argmax(probabilities, axis=1).astype(np.int32, copy=False)
        else:
            predicted_labels = (np.asarray(backdoor_scores) >= 0.0).astype(np.int32, copy=False)

        return SupervisedPredictionOutputs(
            backdoor_scores=np.asarray(backdoor_scores, dtype=np.float64).reshape(-1),
            predicted_labels=np.asarray(predicted_labels, dtype=np.int32).reshape(-1),
            probabilities=probabilities,
            logits=logits,
        )

    observed_classes = _observed_model_classes(model)
    probabilities: np.ndarray | None = None
    logits: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        raw_proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        if raw_proba.ndim == 1:
            raw_proba = raw_proba.reshape(-1, 1)
        probabilities = _align_task_matrix(
            raw_proba,
            observed_classes=observed_classes,
            task_spec=task_spec,
            fill_value=0.0,
        )
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    elif hasattr(model, "decision_function"):
        raw_decision = np.asarray(model.decision_function(x), dtype=np.float64)
        if raw_decision.ndim == 1:
            raw_decision = np.column_stack([-raw_decision, raw_decision])
        logits = _align_task_matrix(
            raw_decision,
            observed_classes=observed_classes,
            task_spec=task_spec,
            fill_value=-math.inf,
        )
        probabilities = _softmax(logits)
    else:
        predicted = np.asarray(model.predict(x), dtype=np.int32).reshape(-1)
        probabilities = np.zeros((predicted.shape[0], task_spec.n_classes), dtype=np.float64)
        probabilities[np.arange(predicted.shape[0], dtype=np.int64), predicted] = 1.0
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))

    if probabilities is None:
        raise RuntimeError("Task-aware prediction expected class probabilities or logits")

    if hasattr(model, "predict"):
        predicted_labels = np.asarray(model.predict(x), dtype=np.int32).reshape(-1)
    else:
        predicted_labels = np.argmax(probabilities, axis=1).astype(np.int32, copy=False)
    backdoor_scores = 1.0 - probabilities[:, task_spec.clean_class_index]
    return SupervisedPredictionOutputs(
        backdoor_scores=np.asarray(backdoor_scores, dtype=np.float64).reshape(-1),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int32).reshape(-1),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        logits=None if logits is None else np.asarray(logits, dtype=np.float64),
    )


def _evaluate_fold(
    *,
    features: np.ndarray | SupervisedFeatureBundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
    task_spec: SupervisedTaskSpec,
    selection_metric_name: str,
    domain_adaptation: dict[str, Any] | None = None,
    rank_values: np.ndarray | None = None,
) -> dict[str, Any]:
    model = create(model_name, params=params, random_state=random_state, task_spec=task_spec)
    valid_features = _slice_supervised_features(features, valid_indices)
    backend = model_backend(model_name)
    if backend == "cnn":
        fit_kwargs: dict[str, Any] = {}
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        if _is_dann_model_name(model_name):
            fit_indices, fit_kwargs = _resolve_dann_fit_inputs(
                manifest={"domain_adaptation": domain_adaptation},
                train_indices=train_indices,
            )
        fit_kwargs.update(
            _resolve_rank_label_fit_kwargs(
                params=params,
                rank_values=rank_values,
                fit_indices=fit_indices,
            )
        )
        train_features = _slice_supervised_features(features, fit_indices)
        model.fit(
            train_features,
            labels[fit_indices],
            validation_data=(valid_features, labels[valid_indices]),
            n_jobs=n_jobs,
            **fit_kwargs,
        )
    else:
        train_features = _slice_supervised_features(features, train_indices)
        model.fit(train_features, labels[train_indices])
    outputs = _predict_task_outputs(model, valid_features, task_spec=task_spec)
    result = {
        "n_train": int(train_indices.size),
        "n_valid": int(valid_indices.size),
        "selection_metric_name": str(selection_metric_name),
    }
    if task_spec.is_binary:
        auc = float(roc_auc_score(labels[valid_indices], outputs.backdoor_scores))
        result["roc_auc"] = auc
        result["binary_auroc"] = auc
        result["selection_metric"] = auc
        return result

    y_valid = np.asarray(labels[valid_indices], dtype=np.int32).reshape(-1)
    y_valid_binary = task_spec.project_known_labels_to_binary(y_valid)
    binary_auroc = float(roc_auc_score(y_valid_binary, outputs.backdoor_scores))
    macro_f1 = float(f1_score(y_valid, outputs.predicted_labels, average="macro"))
    accuracy = float(accuracy_score(y_valid, outputs.predicted_labels))
    support_unique, support_counts = np.unique(y_valid, return_counts=True)
    if selection_metric_name == SELECTION_METRIC_BINARY_AUROC:
        result["selection_metric"] = binary_auroc
    elif selection_metric_name == SELECTION_METRIC_MACRO_F1:
        result["selection_metric"] = macro_f1
    else:
        raise ValueError(
            f"Unsupported selection metric {selection_metric_name!r} for multiclass supervised evaluation"
        )
    result["binary_auroc"] = binary_auroc
    result["macro_f1"] = macro_f1
    result["accuracy"] = accuracy
    result["class_support"] = [
        {
            "class_index": int(class_index),
            "class_name": str(task_spec.class_names[int(class_index)]),
            "count": int(count),
        }
        for class_index, count in zip(support_unique.tolist(), support_counts.tolist())
    ]
    return result


def _evaluate_candidate(
    *,
    features: np.ndarray | SupervisedFeatureBundle,
    labels: np.ndarray,
    task: dict[str, Any],
    cv_split_groups: list[dict[str, Any]],
    n_jobs: int,
    task_spec: SupervisedTaskSpec,
    selection_metric_name: str,
    domain_adaptation: dict[str, Any] | None = None,
    rank_values: np.ndarray | None = None,
) -> dict[str, Any]:
    eval_jobs: list[tuple[int, dict[str, Any]]] = []
    for group in cv_split_groups:
        seed = int(group["random_state"])
        for split in group["cv_splits"]:
            eval_jobs.append((seed, split))

    backend = model_backend(str(task["model_name"]))
    if backend != "sklearn" or n_jobs == 1 or len(eval_jobs) <= 1 or joblib is None:
        evaluated: list[tuple[int, dict[str, Any]]] = []
        for seed, split in eval_jobs:
            tr = np.asarray(split["train_indices"], dtype=np.int64)
            val = np.asarray(split["valid_indices"], dtype=np.int64)
            row = _evaluate_fold(
                features=features,
                labels=labels,
                train_indices=tr,
                valid_indices=val,
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
                n_jobs=n_jobs,
                task_spec=task_spec,
                selection_metric_name=selection_metric_name,
                domain_adaptation=domain_adaptation,
                rank_values=rank_values,
            )
            row["cv_random_state"] = int(seed)
            evaluated.append((seed, row))
    else:
        raw_rows = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_evaluate_fold)(
                features=features,
                labels=labels,
                train_indices=np.asarray(split["train_indices"], dtype=np.int64),
                valid_indices=np.asarray(split["valid_indices"], dtype=np.int64),
                model_name=str(task["model_name"]),
                params=dict(task["params"]),
                random_state=seed,
                n_jobs=n_jobs,
                task_spec=task_spec,
                selection_metric_name=selection_metric_name,
                domain_adaptation=domain_adaptation,
                rank_values=rank_values,
            )
            for seed, split in eval_jobs
        )
        evaluated = []
        for (seed, _), row in zip(eval_jobs, raw_rows):
            row_with_seed = dict(row)
            row_with_seed["cv_random_state"] = int(seed)
            evaluated.append((seed, row_with_seed))

    fold_rows = [row for _, row in evaluated]
    selection_scores = [float(row["selection_metric"]) for row in fold_rows if row.get("selection_metric") is not None]

    seed_results: list[dict[str, Any]] = []
    unique_seeds = sorted({int(seed) for seed, _ in evaluated})
    for seed in unique_seeds:
        seed_fold_rows = [row for seed_value, row in evaluated if int(seed_value) == int(seed)]
        seed_scores = [
            float(row["selection_metric"])
            for row in seed_fold_rows
            if row.get("selection_metric") is not None
        ]
        seed_result = {
            "random_state": int(seed),
            "fold_results": seed_fold_rows,
            "selection_metric_name": selection_metric_name,
            "selection_metric_mean": float(np.mean(seed_scores)) if seed_scores else None,
            "selection_metric_std": float(np.std(seed_scores)) if seed_scores else None,
        }
        if task_spec.is_binary:
            seed_result["roc_auc_mean"] = seed_result["selection_metric_mean"]
            seed_result["roc_auc_std"] = seed_result["selection_metric_std"]
            seed_result["binary_auroc_mean"] = seed_result["selection_metric_mean"]
            seed_result["binary_auroc_std"] = seed_result["selection_metric_std"]
        else:
            binary_auroc_scores = [
                float(row["binary_auroc"]) for row in seed_fold_rows if row.get("binary_auroc") is not None
            ]
            macro_f1_scores = [float(row["macro_f1"]) for row in seed_fold_rows if row.get("macro_f1") is not None]
            accuracy_scores = [float(row["accuracy"]) for row in seed_fold_rows if row.get("accuracy") is not None]
            seed_result["binary_auroc_mean"] = float(np.mean(binary_auroc_scores)) if binary_auroc_scores else None
            seed_result["binary_auroc_std"] = float(np.std(binary_auroc_scores)) if binary_auroc_scores else None
            seed_result["macro_f1_mean"] = float(np.mean(macro_f1_scores)) if macro_f1_scores else None
            seed_result["macro_f1_std"] = float(np.std(macro_f1_scores)) if macro_f1_scores else None
            seed_result["accuracy_mean"] = float(np.mean(accuracy_scores)) if accuracy_scores else None
            seed_result["accuracy_std"] = float(np.std(accuracy_scores)) if accuracy_scores else None
        seed_results.append(seed_result)

    result = {
        "task_index": int(task["task_index"]),
        "model_name": str(task["model_name"]),
        "params": dict(task["params"]),
        "complexity_rank": int(task["complexity_rank"]),
        "normalization_policy": str(task["normalization_policy"]),
        "status": "ok",
        "selection_metric_name": selection_metric_name,
        "fold_results": fold_rows,
        "seed_results": seed_results,
        "selection_metric_mean": float(np.mean(selection_scores)) if selection_scores else None,
        "selection_metric_std": float(np.std(selection_scores)) if selection_scores else None,
    }
    if task_spec.is_binary:
        result["roc_auc_mean"] = result["selection_metric_mean"]
        result["roc_auc_std"] = result["selection_metric_std"]
        result["binary_auroc_mean"] = result["selection_metric_mean"]
        result["binary_auroc_std"] = result["selection_metric_std"]
    else:
        binary_auroc_scores = [float(row["binary_auroc"]) for row in fold_rows if row.get("binary_auroc") is not None]
        macro_f1_scores = [float(row["macro_f1"]) for row in fold_rows if row.get("macro_f1") is not None]
        accuracy_scores = [float(row["accuracy"]) for row in fold_rows if row.get("accuracy") is not None]
        result["binary_auroc_mean"] = float(np.mean(binary_auroc_scores)) if binary_auroc_scores else None
        result["binary_auroc_std"] = float(np.std(binary_auroc_scores)) if binary_auroc_scores else None
        result["macro_f1_mean"] = float(np.mean(macro_f1_scores)) if macro_f1_scores else None
        result["macro_f1_std"] = float(np.std(macro_f1_scores)) if macro_f1_scores else None
        result["accuracy_mean"] = float(np.mean(accuracy_scores)) if accuracy_scores else None
        result["accuracy_std"] = float(np.std(accuracy_scores)) if accuracy_scores else None
    return result


def _task_result_path(task_dir: Path, task_index: int) -> Path:
    return task_dir / f"task_{task_index:04d}.json"


def _compact_extractor_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    drop_keys = {
        "feature_names",
        "incoming_metadata",
        "merge_stats",
        "merged_with_existing_output",
        "merge_existing_output_dir",
        "block_names_raw",
        "base_block_names_raw",
        "qv_sum_block_names_raw",
    }
    extractor_name = str(metadata.get("extractor") or metadata.get("extractor_name") or "")
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in drop_keys or key == "shapes" or key.endswith("_shapes"):
            continue
        if key == "extractor_params" and isinstance(value, dict) and extractor_name == "spectral":
            compact[str(key)] = spectral_extractor_params(value)
            continue
        if isinstance(value, dict):
            nested = _compact_extractor_metadata(value)
            if nested:
                compact[str(key)] = nested
            continue
        compact[str(key)] = value
    return compact


def _score_percentile_ranks(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.float64)
    ranks = np.argsort(np.argsort(scores))
    return np.asarray(ranks / max(1, scores.size - 1), dtype=np.float64)


def _resolve_manifest_path_for_reporting(manifest: dict[str, Any]) -> Path:
    raw_value = str(manifest["manifest_json"])
    resolved = resolve_manifest_path(raw_value)
    if resolved.exists():
        return resolved

    raw_path = Path(raw_value).expanduser()

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        run_dir = Path(str(manifest["run_dir"])).expanduser().resolve()
        if len(run_dir.parents) >= 3:
            candidates.append((run_dir.parents[2] / raw_path).resolve())
        candidates.append(raw_path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_manifest_items_for_tuning_manifest(manifest: dict[str, Any]) -> list[Any]:
    manifest_path = _resolve_manifest_path_for_reporting(manifest)
    feature_loading_mode = str(manifest["data"].get("feature_loading_mode", "materialized"))
    mode = str(manifest["mode"])

    if feature_loading_mode == "external_source":
        if mode == "joint":
            train_items, infer_items = parse_joint_manifest_json_by_model_name(manifest_path=manifest_path)
            return train_items + infer_items
        return parse_single_manifest_json_by_model_name(
            manifest_path=manifest_path,
            section_key="path",
        )

    dataset_root = Path(str(manifest["dataset_root"])).expanduser().resolve()
    if mode == "joint":
        train_items, infer_items = parse_joint_manifest_json(
            manifest_path=manifest_path,
            dataset_root=dataset_root,
        )
        return train_items + infer_items
    return parse_single_manifest_json(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        section_key="path",
    )


def _build_threshold_specs(train_scores: np.ndarray, percentiles: list[float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for pct in percentiles:
        specs.append(
            {
                "percentile_from_train": float(pct),
                "threshold": float(np.percentile(train_scores, float(pct))),
            }
        )
    return specs


def _evaluate_attack_threshold_rows(
    *,
    group_scores: np.ndarray,
    known_mask: np.ndarray,
    known_labels: np.ndarray | None,
    threshold_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold_rows: list[dict[str, Any]] = []
    for spec in threshold_specs:
        threshold = float(spec["threshold"])
        flagged = group_scores >= threshold
        n_flagged = int(np.sum(flagged))
        row: dict[str, Any] = {key: value for key, value in spec.items() if key != "threshold"}
        row.update(
            {
                "threshold": threshold,
                "n_flagged": n_flagged,
                "fraction_flagged": float(n_flagged / max(1, group_scores.size)),
            }
        )

        if known_labels is not None:
            known_flagged = flagged[known_mask]
            positives = int(np.sum(known_labels == 1))
            negatives = int(np.sum(known_labels == 0))
            tp = int(np.sum((known_labels == 1) & known_flagged))
            fp = int(np.sum((known_labels == 0) & known_flagged))
            if n_flagged > 0:
                row["precision"] = float(tp / n_flagged)
            if positives > 0:
                row["recall"] = float(tp / positives)
            if negatives > 0:
                row["false_positive_rate"] = float(fp / negatives)

        threshold_rows.append(row)

    return threshold_rows


def _summarize_attack_slice(
    *,
    group_scores: np.ndarray,
    group_ranks: np.ndarray,
    group_labels: list[int | None],
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None,
    source_subsets: list[str],
) -> dict[str, Any]:
    clean_count = sum(1 for label in group_labels if label == 0)
    backdoored_count = sum(1 for label in group_labels if label == 1)
    unknown_count = sum(1 for label in group_labels if label is None)

    known_mask = np.asarray([lbl is not None for lbl in group_labels], dtype=bool)
    known_labels: np.ndarray | None = None
    if bool(np.any(known_mask)):
        known_labels = np.asarray(
            [int(group_labels[i]) for i in range(len(group_labels)) if known_mask[i]],
            dtype=np.int32,
        )
    known_scores = np.asarray(group_scores[known_mask], dtype=np.float64)

    summary = {
        "n_samples": int(group_scores.size),
        "source_subsets": list(source_subsets),
        "label_counts": {
            "clean": int(clean_count),
            "backdoored": int(backdoored_count),
            "unknown": int(unknown_count),
        },
        "score_summary": summarize_scores(group_scores),
        "score_percentile_rank_summary": summarize_scores(group_ranks),
        "offline_metrics": compute_offline_metrics(known_labels, known_scores),
        "threshold_evaluation": _evaluate_attack_threshold_rows(
            group_scores=group_scores,
            known_mask=known_mask,
            known_labels=known_labels,
            threshold_specs=threshold_specs,
        ),
    }
    if selected_threshold_specs:
        summary["selected_threshold_evaluation"] = _evaluate_attack_threshold_rows(
            group_scores=group_scores,
            known_mask=known_mask,
            known_labels=known_labels,
            threshold_specs=selected_threshold_specs,
        )
    return summary


def _summarize_attack_groups(
    *,
    sample_identities: list[AttackSampleIdentity],
    labels: list[int | None],
    scores: np.ndarray,
    threshold_specs: list[dict[str, Any]],
    selected_threshold_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(sample_identities) != len(labels) or len(sample_identities) != int(scores.size):
        raise ValueError("Attack analysis inputs must have the same length")

    score_ranks = _score_percentile_ranks(np.asarray(scores, dtype=np.float64))
    all_indices_by_attack: dict[str, list[int]] = {}
    positive_by_attack: dict[str, list[int]] = {}
    unknown_by_attack: dict[str, list[int]] = {}
    attack_subset_names: dict[str, set[str]] = {}
    clean_indices: list[int] = []
    clean_subset_names: set[str] = set()

    for idx, (identity, label) in enumerate(zip(sample_identities, labels)):
        all_indices_by_attack.setdefault(identity.attack_name, []).append(int(idx))
        attack_subset_names.setdefault(identity.attack_name, set()).add(identity.subset_name)
        if label == 1:
            positive_by_attack.setdefault(identity.attack_name, []).append(int(idx))
        elif label == 0:
            clean_indices.append(int(idx))
            clean_subset_names.add(identity.subset_name)
        else:
            unknown_by_attack.setdefault(identity.attack_name, []).append(int(idx))

    grouped_indices: dict[str, list[int]] = {}
    if positive_by_attack:
        for attack in sorted(positive_by_attack):
            combined = (
                list(positive_by_attack[attack])
                + list(clean_indices)
                + list(unknown_by_attack.get(attack, []))
            )
            grouped_indices[attack] = sorted(set(int(i) for i in combined))
    else:
        grouped_indices = {
            attack: sorted(int(i) for i in idx_list)
            for attack, idx_list in all_indices_by_attack.items()
        }

    attacks: dict[str, Any] = {}
    for attack in sorted(grouped_indices):
        idx = np.asarray(grouped_indices[attack], dtype=np.int64)
        attacks[attack] = _summarize_attack_slice(
            group_scores=np.asarray(scores[idx], dtype=np.float64),
            group_ranks=np.asarray(score_ranks[idx], dtype=np.float64),
            group_labels=[labels[int(i)] for i in idx.tolist()],
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
            source_subsets=sorted(attack_subset_names.get(attack, set())),
        )

    return {
        "grouping_rule": (
            "one-vs-clean per attack using a single shared clean pool built from every label0 sample; "
            "known PADBench attacks are canonicalized by name (RIPPLE, syntactic, insertsent, stybkd); "
            "all other folders contribute label1 samples under the folder-derived attack name after "
            "removing model/config tokens"
        ),
        "clean_pool": {
            "n_samples": int(len(clean_indices)),
            "source_subsets": sorted(str(x) for x in clean_subset_names),
        },
        "n_attacks": int(len(attacks)),
        "attacks": attacks,
    }


def _context_from_run_dir(run_dir: Path) -> RunContext:
    output_root = run_dir.parents[1] if len(run_dir.parents) >= 2 else run_dir.parent
    features_dir = run_dir / "features"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"

    for path in [features_dir, models_dir, reports_dir, plots_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_dir.name,
        run_dir=run_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
    )


def _prepare_supervised_run(
    *,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
    spectral_features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float],
    feature_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    slurm_max_concurrent: str,
    slurm_cpus_per_task: str,
    task_mode: str,
    multiclass_attack_names: list[str] | None,
    cnn_hyperparams: Path | None,
    autoencoder_hyperparams: Path | None,
    dann_source_rank: int,
    dann_target_adaptation_percent: int,
    dann_lambda_max: float,
    dann_lambda_gamma: float,
    dann_lr_alpha: float,
    dann_lr_beta: float,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    selection_metric: str | None = None,
) -> dict[str, Any]:
    manifest_json = resolve_manifest_path(manifest_json)
    if not manifest_json.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_json}")
    if feature_file is None:
        raise ValueError(
            "Supervised pipeline requires --feature-file. Run feature extraction first, then pass the "
            "extracted feature bundle into supervised."
        )
    if not spectral_features:
        raise ValueError(
            "Supervised pipeline requires --features. Specify the feature groups to select from the "
            "extracted feature bundle."
        )
    if cnn_hyperparams is not None and model_name not in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME, "all"}:
        raise ValueError("--cnn-hyperparams is only supported when --model is cnn_1d, cnn_1d_dann, or all")
    if autoencoder_hyperparams is not None and model_name not in {"autoencoder_knn", "all"}:
        raise ValueError("--autoencoder-hyperparams is only supported when --model is autoencoder_knn or all")
    if bool(class_weight_loss) and bool(rank_label_weight_loss):
        raise ValueError("--class-weight-loss and --rank-label-weight-loss are mutually exclusive")
    task_spec = _resolve_supervised_task_spec(
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
    )
    selection_metric_name = _resolve_selection_metric(
        selection_metric,
        task_spec=task_spec,
    )

    mode = _detect_manifest_mode(manifest_json)
    if mode == "joint":
        train_items, infer_items = parse_joint_manifest_json_by_model_name(
            manifest_path=manifest_json,
        )
        all_items = train_items + infer_items
        train_indices = np.arange(0, len(train_items), dtype=np.int64)
        infer_indices = np.arange(len(train_items), len(all_items), dtype=np.int64)
    else:
        all_items = parse_single_manifest_json_by_model_name(
            manifest_path=manifest_json,
            section_key="path",
        )
        train_items = all_items
        infer_items = []
        train_indices = np.arange(0, len(all_items), dtype=np.int64)
        infer_indices = np.asarray([], dtype=np.int64)

    if not train_items:
        raise ValueError("No training items resolved from manifest")
    all_sample_identities = infer_attack_sample_identities(all_items)

    train_split_percent = _resolve_train_split_percent(train_split_percent)
    calibration_split_percent = _resolve_calibration_split_percent(calibration_split_percent)
    accepted_fprs = _resolve_accepted_fprs(accepted_fpr)
    if (calibration_split_percent is None) != (accepted_fprs is None):
        raise ValueError("--calibration-split and --accepted-fpr must either both be set or both be omitted")

    ctx = create_run_context(
        pipeline=PIPELINE_NAME,
        output_root=output_root,
        run_id=run_id,
    )

    selected_spectral_features = list(spectral_features)
    feature_params = {
        "dtype": dtype_name,
        "block_size": int(stream_block_size),
        "spectral_features": list(selected_spectral_features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": str(spectral_moment_source),
        "spectral_qv_sum_mode": str(spectral_qv_sum_mode),
        "spectral_entrywise_delta_mode": str(spectral_entrywise_delta_mode),
    }
    extractor_warnings: list[str] = []
    feature_loading_mode = "external_source"
    resolved_bundle_paths = _resolve_supervised_feature_bundle_paths(feature_file)
    features, external_metadata, external_warnings, selected_expected_indices = _load_external_spectral_bundle(
        feature_file=resolved_bundle_paths.feature_path,
        model_names_file=resolved_bundle_paths.model_names_path,
        metadata_file=resolved_bundle_paths.metadata_path,
        group_mask_file=resolved_bundle_paths.group_mask_path,
        value_mask_file=resolved_bundle_paths.value_mask_path,
        group_names_file=resolved_bundle_paths.group_names_path,
        expected_model_names=[item.model_name for item in all_items],
        spectral_features=selected_spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        allow_manifest_subset=True,
    )
    extractor_warnings.extend(external_warnings)

    if int(selected_expected_indices.size) != int(len(all_items)):
        all_items = [all_items[int(i)] for i in selected_expected_indices.tolist()]
        all_sample_identities = [all_sample_identities[int(i)] for i in selected_expected_indices.tolist()]
        train_indices = _compact_indices_to_selected_scope(
            train_indices,
            selected_expected_indices=selected_expected_indices,
        )
        infer_indices = _compact_indices_to_selected_scope(
            infer_indices,
            selected_expected_indices=selected_expected_indices,
        )
        train_items = [all_items[int(i)] for i in train_indices.tolist()]
        infer_items = [all_items[int(i)] for i in infer_indices.tolist()]
        if not train_items:
            raise ValueError(
                "No training items remain after intersecting the manifest with the available "
                "external feature/model-name set"
            )

    run_metadata_path = ctx.features_dir / "spectral_metadata.json"
    extractor_metadata = {
        **external_metadata,
        "external_feature_source": str(resolved_bundle_paths.feature_path),
        "external_model_names_source": str(resolved_bundle_paths.model_names_path),
        "external_metadata_source": (
            str(resolved_bundle_paths.metadata_path) if resolved_bundle_paths.metadata_path is not None else None
        ),
        "external_group_mask_source": (
            str(resolved_bundle_paths.group_mask_path)
            if resolved_bundle_paths.group_mask_path is not None
            else None
        ),
        "external_value_mask_source": (
            str(resolved_bundle_paths.value_mask_path)
            if resolved_bundle_paths.value_mask_path is not None
            else None
        ),
        "external_group_names_source": (
            str(resolved_bundle_paths.group_names_path)
            if resolved_bundle_paths.group_names_path is not None
            else None
        ),
        "loaded_external_features": True,
    }
    external_dataset_reference_payload = (
        resolve_dataset_reference_for_metadata(resolved_bundle_paths.metadata_path)
        if resolved_bundle_paths.metadata_path is not None
        else None
    )
    write_spectral_metadata(
        run_metadata_path,
        internal_metadata=extractor_metadata,
        dataset_layouts=dataset_layouts_from_source(
            metadata=extractor_metadata,
            dataset_reference_payload=external_dataset_reference_payload,
        ),
    )

    artifacts = {
        "feature_path": str(resolved_bundle_paths.feature_path),
        "labels_path": None,
        "model_names_path": None,
        "metadata_path": str(run_metadata_path),
    }
    representation_kind = _normalized_representation_kind(
        external_metadata,
        feature_ndim=(
            int(features.values.ndim) if isinstance(features, SupervisedFeatureBundle) else int(features.ndim)
        ),
    )
    model_names, incompatible_model_names = _compatible_model_names_for_representation(
        requested_model_name=model_name,
        representation_kind=representation_kind,
    )
    if model_name == "all" and incompatible_model_names:
        extractor_warnings.append(
            "Filtered incompatible supervised models for representation_kind="
            f"{representation_kind!r}: skipped {incompatible_model_names}"
        )
    cnn_model_selected = any(name in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME} for name in model_names)
    cnn_rank_label_weight_loss = bool(rank_label_weight_loss and cnn_model_selected)
    if bool(rank_label_weight_loss) and not cnn_model_selected:
        extractor_warnings.append(
            "Ignored --rank-label-weight-loss because no CNN model is selected for this supervised run"
        )
    resolved_cnn_hyperparam_axes: dict[str, list[Any]] | None = None
    cnn_hyperparams_info: dict[str, Any] | None = None
    resolved_autoencoder_hyperparam_axes: dict[str, list[Any]] | None = None
    autoencoder_hyperparams_info: dict[str, Any] | None = None
    if cnn_model_selected:
        resolved_cnn_hyperparam_axes, cnn_hyperparams_info = resolve_cnn_hyperparams(cnn_hyperparams)
    elif cnn_hyperparams is not None:
        extractor_warnings.append(
            "Ignored --cnn-hyperparams because no CNN model is selected for this supervised run"
        )
    autoencoder_model_selected = any(name == "autoencoder_knn" for name in model_names)
    if autoencoder_model_selected:
        resolved_autoencoder_hyperparam_axes, autoencoder_hyperparams_info = resolve_autoencoder_hyperparams(
            autoencoder_hyperparams
        )
    elif autoencoder_hyperparams is not None:
        extractor_warnings.append(
            "Ignored --autoencoder-hyperparams because no autoencoder_knn model is selected for this supervised run"
        )

    labels_values, labels_known, labels_raw = _labels_from_items(
        all_items,
        task_spec=task_spec,
        sample_identities=all_sample_identities,
    )
    split_warnings: list[str] = []
    if mode == "single":
        if train_split_percent < 100 and not bool(np.all(labels_known)):
            raise ValueError(
                "Single-manifest --train-split requires labels for every sample in the manifest"
            )
        if split_by_folder:
            train_indices, infer_indices, split_warnings, split_summary = _build_single_manifest_folder_label_split(
                items=all_items,
                labels=labels_values,
                train_split_percent=train_split_percent,
                random_state=random_state,
            )
        else:
            train_indices, infer_indices, split_warnings, split_summary = _build_single_manifest_stratified_split(
                labels=labels_values,
                train_split_percent=train_split_percent,
                random_state=random_state,
            )
    else:
        if train_split_percent != 100:
            raise ValueError(
                "--train-split values below 100 are only supported for single manifests; "
                "joint manifests already define train and inference partitions"
            )
        split_summary = {
            "strategy": "manifest_defined",
            "split_by_folder": False,
            "requested_train_split_percent": int(train_split_percent),
            "random_state": int(random_state),
            "n_train": int(train_indices.size),
            "n_inference": int(infer_indices.size),
            "train_label_counts": _label_count_rows(labels_values[train_indices]),
            "inference_label_counts": _label_count_rows(labels_values[infer_indices][labels_known[infer_indices]]),
            "n_inference_unknown_label": int(np.sum(~labels_known[infer_indices])),
        }

    train_known = labels_known[train_indices]
    if not bool(np.all(train_known)):
        raise ValueError("Training samples must all have known labels for supervised learning")

    train_pool_indices = np.asarray(train_indices, dtype=np.int64)
    fit_train_indices = np.asarray(train_pool_indices, dtype=np.int64)
    domain_adaptation_config, domain_warnings = _build_dann_domain_adaptation_config(
        model_names=model_names,
        all_items=all_items,
        train_indices=train_pool_indices,
        infer_indices=infer_indices,
        labels_values=labels_values,
        labels_known=labels_known,
        mode=mode,
        source_rank=int(dann_source_rank),
        target_adaptation_percent=int(dann_target_adaptation_percent),
        lambda_max=float(dann_lambda_max),
        lambda_gamma=float(dann_lambda_gamma),
        lr_alpha=float(dann_lr_alpha),
        lr_beta=float(dann_lr_beta),
        random_state=int(random_state),
        output_dir=ctx.features_dir,
    )
    if domain_adaptation_config is not None:
        if isinstance(split_summary, dict):
            split_summary = dict(split_summary)
            split_summary["strategy"] = "manifest_defined_with_dann_rank_adversarial_zero_shot"
            split_summary["n_train"] = int(train_pool_indices.size)
            split_summary["n_inference"] = int(infer_indices.size)
            split_summary["train_ranks"] = list(domain_adaptation_config["train_ranks"])
            split_summary["target_train_ranks"] = list(domain_adaptation_config["target_train_ranks"])
            split_summary["zero_shot_inference_ranks"] = list(
                domain_adaptation_config["zero_shot_inference_ranks"]
            )
            split_summary["train_label_counts"] = _label_count_rows(labels_values[train_pool_indices])
            split_summary["inference_label_counts"] = _label_count_rows(
                labels_values[infer_indices][labels_known[infer_indices]]
            )

    calibration_indices = np.asarray([], dtype=np.int64)
    calibration_summary: dict[str, Any] | None = None
    calibration_warnings: list[str] = []
    if calibration_split_percent is not None:
        if domain_adaptation_config is not None:
            all_rank_values = np.asarray(
                [_parse_rank_from_model_name(item.model_name) for item in all_items],
                dtype=np.int64,
            )
            rank_label_warning = _rank_label_stratification_warning(
                candidate_indices=train_pool_indices,
                ranks=all_rank_values,
                labels=labels_values,
                split_name="calibration",
            )
            if rank_label_warning is None:
                fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = _build_calibration_rank_label_split(
                    candidate_indices=train_pool_indices,
                    ranks=all_rank_values,
                    labels=labels_values,
                    calibration_split_percent=calibration_split_percent,
                    random_state=random_state,
                )
            else:
                calibration_warnings.append(rank_label_warning)
                fit_train_indices, calibration_indices, fallback_warnings, calibration_summary = _build_calibration_stratified_split(
                    candidate_indices=train_pool_indices,
                    labels=labels_values,
                    calibration_split_percent=calibration_split_percent,
                    random_state=random_state,
                )
                calibration_warnings.extend(fallback_warnings)
        elif split_by_folder:
            fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = _build_calibration_folder_label_split(
                items=all_items,
                candidate_indices=train_pool_indices,
                labels=labels_values,
                calibration_split_percent=calibration_split_percent,
                random_state=random_state,
            )
        else:
            fit_train_indices, calibration_indices, calibration_warnings, calibration_summary = _build_calibration_stratified_split(
                candidate_indices=train_pool_indices,
                labels=labels_values,
                calibration_split_percent=calibration_split_percent,
                random_state=random_state,
            )

    fit_train_labels = labels_values[fit_train_indices]
    requested_states = list(cv_random_states) if cv_random_states else [int(random_state)]
    dedup_states: list[int] = []
    seen_states: set[int] = set()
    for state in requested_states:
        state_int = int(state)
        if state_int in seen_states:
            continue
        seen_states.add(state_int)
        dedup_states.append(state_int)
    if not dedup_states:
        dedup_states = [int(random_state)]

    tasks: list[dict[str, Any]] = []
    task_index = 0
    for selected_model in model_names:
        complexity = model_complexity_rank(selected_model)
        model_candidate_params = candidate_params(
            selected_model,
            cnn_hyperparams=(
                resolved_cnn_hyperparam_axes
                if selected_model in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME}
                else None
            ),
            autoencoder_hyperparams=(
                resolved_autoencoder_hyperparam_axes if selected_model == "autoencoder_knn" else None
            ),
        )
        for params in model_candidate_params:
            task_params = dict(params)
            if selected_model in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME} and bool(class_weight_loss):
                task_params["class_weight_loss"] = True
            if selected_model in {CNN_1D_MODEL_NAME, CNN_1D_DANN_MODEL_NAME} and bool(cnn_rank_label_weight_loss):
                task_params["rank_label_weight_loss"] = True
            if selected_model == CNN_1D_DANN_MODEL_NAME:
                task_params.update(
                    {
                        "source_rank": int(dann_source_rank),
                        "dann_lambda_max": float(dann_lambda_max),
                        "dann_lambda_gamma": float(dann_lambda_gamma),
                    }
                )
            tasks.append(
                {
                    "task_index": int(task_index),
                    "model_name": selected_model,
                    "params": task_params,
                    "complexity_rank": int(complexity),
                    "normalization_policy": normalization_policy(selected_model),
                }
            )
            task_index += 1

    _validated_train_label_counts(fit_train_labels)
    execution_mode = "singleton_no_cv" if len(tasks) == 1 else "cross_validation"
    cv_split_groups: list[dict[str, Any]] = []
    cv_splits: list[dict[str, Any]] = []
    if execution_mode == "singleton_no_cv":
        cv_folds_resolved = 0
        cv_warnings = [
            "Skipped cross-validation because tuning search contains a single candidate; "
            "singleton_no_cv execution mode will fit only during finalize"
        ]
        cv_stratification_summary = {
            "strategy": None,
            "split_by_folder": False,
            "split_by_rank_label": False,
        }
        estimated_total_fits = 1
    else:
        cv_stratification_labels = fit_train_labels
        cv_stratification_summary = {
            "strategy": "label",
            "split_by_folder": False,
            "split_by_rank_label": False,
        }
        cv_warnings: list[str] = []
        if split_by_folder:
            folder_label_warning = _folder_label_stratification_warning(
                items=all_items,
                candidate_indices=fit_train_indices,
                labels=labels_values,
                split_name="cross-validation",
            )
            if folder_label_warning is None:
                cv_stratification_labels = _folder_label_stratification_keys(
                    items=all_items,
                    candidate_indices=fit_train_indices,
                    labels=labels_values,
                )
                cv_stratification_summary = {
                    "strategy": "folder_label",
                    "split_by_folder": True,
                    "split_by_rank_label": False,
                    "folder_count": int(
                        len(
                            {
                                _split_folder_name(all_items[int(idx)])
                                for idx in fit_train_indices.tolist()
                            }
                        )
                    ),
                }
            else:
                cv_warnings.append(folder_label_warning)
        elif domain_adaptation_config is not None:
            all_rank_values = np.asarray(
                [_parse_rank_from_model_name(item.model_name) for item in all_items],
                dtype=np.int64,
            )
            rank_label_warning = _rank_label_stratification_warning(
                candidate_indices=fit_train_indices,
                ranks=all_rank_values,
                labels=labels_values,
                split_name="cross-validation",
            )
            if rank_label_warning is None:
                cv_stratification_labels = _rank_label_stratification_keys(
                    candidate_indices=fit_train_indices,
                    ranks=all_rank_values,
                    labels=labels_values,
                )
                cv_stratification_summary = {
                    "strategy": "rank_label",
                    "split_by_folder": False,
                    "split_by_rank_label": True,
                }
            else:
                cv_warnings.append(rank_label_warning)
        cv_folds_resolved, fold_warnings = _sanitize_cv_folds(cv_stratification_labels, cv_folds)
        cv_warnings.extend(fold_warnings)
        for state in dedup_states:
            cv_split_groups.append(
                {
                    "random_state": int(state),
                    "cv_splits": _build_cv_splits(
                        train_indices=fit_train_indices,
                        train_labels=cv_stratification_labels,
                        cv_folds=cv_folds_resolved,
                        random_state=int(state),
                    ),
                }
            )
        cv_splits = list(cv_split_groups[0]["cv_splits"])
        estimated_total_fits = _estimate_total_fit_count(
            n_tasks=len(tasks),
            cv_split_groups=cv_split_groups,
        )
    grid_warnings = _grid_search_warnings(
        n_tasks=len(tasks),
        estimated_total_fits=estimated_total_fits,
    )

    labels_value_path = ctx.features_dir / "supervised_label_values.npy"
    labels_known_path = ctx.features_dir / "supervised_label_known.npy"
    np.save(labels_value_path, labels_values)
    np.save(labels_known_path, labels_known.astype(np.int8))

    model_names_path = ctx.features_dir / "supervised_model_names.json"
    with open(model_names_path, "w", encoding="utf-8") as f:
        json.dump([item.model_name for item in all_items], f, indent=2)

    rank_values_path: Path | None = None
    if bool(cnn_rank_label_weight_loss):
        rank_values = np.asarray(
            [_parse_rank_from_model_name(item.model_name) for item in all_items],
            dtype=np.int64,
        )
        rank_values_path = ctx.features_dir / "supervised_rank_values.npy"
        np.save(rank_values_path, rank_values)

    slurm_cpus = resolve_slurm_cpus_per_task(slurm_cpus_per_task)
    slurm_concurrency = resolve_slurm_max_concurrent(slurm_max_concurrent, slurm_cpus)

    task_dir = ctx.reports_dir / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    tuning_manifest = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(ctx.run_dir),
        "mode": mode,
        "manifest_json": str(manifest_json.expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "task": task_spec.to_dict(),
        "data": {
            "n_samples": int(_supervised_feature_row_count(features)),
            "train_indices": [int(x) for x in fit_train_indices.tolist()],
            "train_pool_indices": [int(x) for x in train_pool_indices.tolist()],
            "calibration_indices": [int(x) for x in calibration_indices.tolist()],
            "infer_indices": [int(x) for x in infer_indices.tolist()],
            "split": split_summary,
            "calibration_split": calibration_summary,
            "feature_path": str(Path(artifacts["feature_path"])),
            "feature_loading_mode": feature_loading_mode,
            "labels_value_path": str(labels_value_path),
            "labels_known_path": str(labels_known_path),
            "model_names_path": str(model_names_path),
            "rank_values_path": str(rank_values_path) if rank_values_path is not None else None,
        },
        "extractor": {
            "name": "spectral",
            "params": feature_params,
            "metadata": extractor_metadata,
            "warnings": extractor_warnings,
            "metadata_path": str(Path(artifacts["metadata_path"])),
        },
        "domain_adaptation": domain_adaptation_config,
        "tuning": {
            "executor": tuning_executor,
            "model_name": model_name,
            "model_names": model_names,
            "metric": str(selection_metric_name),
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
            "train_split_percent": int(train_split_percent),
            "split_by_folder": bool(split_by_folder),
            "class_weight_loss": bool(class_weight_loss),
            "rank_label_weight_loss": bool(cnn_rank_label_weight_loss),
            "calibration_split_percent": (
                int(calibration_split_percent) if calibration_split_percent is not None else None
            ),
            "cv_random_states": [int(x) for x in dedup_states],
            "cv_folds_requested": int(cv_folds),
            "cv_folds_resolved": int(cv_folds_resolved),
            "execution_mode": execution_mode,
            "estimated_total_fits": int(estimated_total_fits),
            "cv_stratification": cv_stratification_summary,
            "cnn_hyperparams": cnn_hyperparams_info,
            "autoencoder_hyperparams": autoencoder_hyperparams_info,
            "cv_splits": cv_splits,
            "cv_split_groups": cv_split_groups,
            "tasks": tasks,
        },
        "threshold_selection": {
            "method": (
                "maximize_recall_subject_to_fpr"
                if calibration_split_percent is not None
                else None
            ),
            "calibration_split_percent": (
                int(calibration_split_percent) if calibration_split_percent is not None else None
            ),
            "accepted_fprs": list(accepted_fprs) if accepted_fprs is not None else None,
            "accepted_fpr": (
                float(accepted_fprs[0])
                if accepted_fprs is not None and len(accepted_fprs) == 1
                else None
            ),
            "split_by_folder": bool(split_by_folder and calibration_split_percent is not None),
        },
        "runtime": {
            "slurm_partition": slurm_partition,
            "slurm_max_concurrent": int(slurm_concurrency),
            "slurm_cpus_per_task": int(slurm_cpus),
            "score_percentiles": [float(x) for x in score_percentiles],
        },
        "warnings": (
            cv_warnings
            + split_warnings
            + calibration_warnings
            + domain_warnings
            + list(extractor_warnings)
            + grid_warnings
        ),
        "labels_preview": labels_raw,
    }

    tuning_manifest_path = ctx.reports_dir / "tuning_manifest.json"
    with open(tuning_manifest_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(tuning_manifest), f, indent=2)

    return {
        "run_dir": str(ctx.run_dir),
        "tuning_manifest": str(tuning_manifest_path),
        "n_tasks": len(tasks),
        "task_dir": str(task_dir),
        "slurm_partition": slurm_partition,
        "slurm_max_concurrent": int(slurm_concurrency),
        "slurm_cpus_per_task": int(slurm_cpus),
        "execution_mode": execution_mode,
        "warnings": tuning_manifest["warnings"],
    }


def _run_supervised_worker(
    *,
    run_dir: Path,
    task_index: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = manifest["tuning"]["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index={task_index} out of range [0, {len(tasks) - 1}]")

    task = tasks[task_index]
    task_spec = _task_spec_from_manifest(manifest)
    tuning_cfg = manifest["tuning"]
    selection_metric_name = _resolve_selection_metric(
        tuning_cfg.get("metric"),
        task_spec=task_spec,
    )
    execution_mode = str(tuning_cfg.get("execution_mode", "cross_validation"))
    if execution_mode == "singleton_no_cv":
        start = perf_counter()
        result = {
            "task_index": int(task["task_index"]),
            "model_name": str(task["model_name"]),
            "params": dict(task["params"]),
            "complexity_rank": int(task["complexity_rank"]),
            "normalization_policy": str(task["normalization_policy"]),
            "status": "ok",
            "execution_mode": execution_mode,
            "selection_metric_name": str(selection_metric_name),
            "selection_metric_mean": None,
            "selection_metric_std": None,
            "fold_results": [],
            "seed_results": [],
            "elapsed_seconds": float(perf_counter() - start),
        }
        if task_spec.is_binary:
            result["roc_auc_mean"] = None
            result["roc_auc_std"] = None
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None
        else:
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None
            result["macro_f1_mean"] = None
            result["macro_f1_std"] = None
            result["accuracy_mean"] = None
            result["accuracy_std"] = None

        task_dir = run_dir / "reports" / "tuning_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = _task_result_path(task_dir, task_index)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_ready(result), f, indent=2)

        return {
            "run_dir": str(run_dir),
            "task_index": int(task_index),
            "result_path": str(out_path),
            "status": result.get("status"),
        }

    features = _load_features_for_tuning_manifest(manifest)
    labels = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
    rank_values = None
    rank_values_path = manifest["data"].get("rank_values_path")
    if isinstance(rank_values_path, str) and rank_values_path:
        rank_values = np.load(rank_values_path).astype(np.int64)
    if "cv_split_groups" in tuning_cfg:
        cv_split_groups = list(tuning_cfg["cv_split_groups"])
    else:
        cv_split_groups = [
            {
                "random_state": int(tuning_cfg["random_state"]),
                "cv_splits": list(tuning_cfg["cv_splits"]),
            }
        ]
    resolved_n_jobs = int(n_jobs) if n_jobs is not None else int(manifest["tuning"]["n_jobs"])

    start = perf_counter()
    try:
        result = _evaluate_candidate(
            features=features,
            labels=labels,
            task=task,
            cv_split_groups=cv_split_groups,
            n_jobs=resolved_n_jobs,
            task_spec=task_spec,
            selection_metric_name=selection_metric_name,
            domain_adaptation=manifest.get("domain_adaptation"),
            rank_values=rank_values,
        )
    except Exception as exc:  # pragma: no cover - failure path asserted via output file shape
        result = {
            "task_index": int(task["task_index"]),
            "model_name": str(task["model_name"]),
            "params": dict(task["params"]),
            "complexity_rank": int(task["complexity_rank"]),
            "normalization_policy": str(task["normalization_policy"]),
            "status": "error",
            "error": str(exc),
            "execution_mode": execution_mode,
            "selection_metric_name": str(selection_metric_name),
            "selection_metric_mean": None,
            "selection_metric_std": None,
            "fold_results": [],
            "seed_results": [],
        }
        if task_spec.is_binary:
            result["roc_auc_mean"] = None
            result["roc_auc_std"] = None
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None
        else:
            result["binary_auroc_mean"] = None
            result["binary_auroc_std"] = None
            result["macro_f1_mean"] = None
            result["macro_f1_std"] = None
            result["accuracy_mean"] = None
            result["accuracy_std"] = None

    result["elapsed_seconds"] = float(perf_counter() - start)

    task_dir = run_dir / "reports" / "tuning_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = _task_result_path(task_dir, task_index)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2)

    return {
        "run_dir": str(run_dir),
        "task_index": int(task_index),
        "result_path": str(out_path),
        "status": result.get("status"),
    }


def _select_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in candidates
        if row.get("status") == "ok" and row.get("selection_metric_mean") is not None
    ]
    if not valid:
        singleton_no_cv = [
            row
            for row in candidates
            if row.get("status") == "ok" and row.get("execution_mode") == "singleton_no_cv"
        ]
        if len(singleton_no_cv) == 1:
            return singleton_no_cv[0]
        raise RuntimeError("No successful tuning candidates available to select a winner")

    ranked = sorted(
        valid,
        key=lambda row: (
            -float(row["selection_metric_mean"]),
            (
                float(row["selection_metric_std"])
                if row.get("selection_metric_std") is not None
                else float("inf")
            ),
            int(row.get("complexity_rank", 10**9)),
            int(row["task_index"]),
        ),
    )
    return ranked[0]


def _save_model(model: Any, path: Path) -> None:
    if hasattr(model, "save"):
        model.save(path)
        return
    if joblib is not None:
        joblib.dump(model, path)
        return
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _winner_feature_weights_mode(
    *,
    skip_feature_importance: bool,
    winner_model_name: str,
    task_spec: SupervisedTaskSpec,
) -> str:
    if skip_feature_importance:
        return "skipped"
    if not task_spec.is_binary:
        return "unsupported_for_task_mode"
    if model_backend(winner_model_name) != "sklearn":
        return "unsupported_for_backend"
    return "pending"


def _finalize_state_path(run_dir: Path) -> Path:
    return run_dir / "reports" / FINALIZE_STATE_FILENAME


def _write_finalize_state(
    *,
    run_dir: Path,
    run_config: dict[str, Any],
    artifacts: dict[str, str | None],
    winner_export_train_features: Path | None,
    winner_export_train_labels: Path | None,
    random_state: int,
    skip_feature_importance: bool,
) -> Path:
    state_path = _finalize_state_path(run_dir)
    payload = {
        "run_config": run_config,
        "artifacts": artifacts,
        "winner_export": {
            "train_feature_matrix_path": (
                str(winner_export_train_features) if winner_export_train_features is not None else None
            ),
            "train_labels_path": (
                str(winner_export_train_labels) if winner_export_train_labels is not None else None
            ),
            "random_state": int(random_state),
            "skip_feature_importance": bool(skip_feature_importance),
        },
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return state_path


def _load_finalize_state(run_dir: Path) -> dict[str, Any]:
    path = _finalize_state_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Finalize state not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in finalize state, got {type(payload).__name__}")
    return payload


def _load_run_config(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_config.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in run config, got {type(payload).__name__}")
    return payload


def _already_finalized_winner_feature_weights_mode(run_dir: Path) -> str | None:
    if _finalize_state_path(run_dir).exists():
        return None
    run_config = _load_run_config(run_dir)
    if run_config is None:
        return None
    mode = str(run_config.get("winner_feature_weights_mode", ""))
    if mode in {"skipped", "unsupported_for_backend", "unsupported_for_task_mode"}:
        return mode
    return None


def _completed_supervised_result(run_dir: Path) -> dict[str, Any]:
    artifact_index_path = run_dir / "artifact_index.json"
    if not artifact_index_path.exists():
        raise FileNotFoundError(f"Artifact index not found: {artifact_index_path}")
    with open(artifact_index_path, "r", encoding="utf-8") as f:
        artifact_index = json.load(f)
    if not isinstance(artifact_index, dict):
        raise ValueError(f"Expected JSON object in artifact index, got {type(artifact_index).__name__}")
    return {
        "run_dir": str(run_dir),
        "report": str(artifact_index.get("report")),
        "train_scores_csv": str(artifact_index.get("train_scores_csv")),
        "inference_scores_csv": (
            str(artifact_index.get("inference_scores_csv"))
            if artifact_index.get("inference_scores_csv")
            else None
        ),
        "best_model": str(artifact_index.get("best_model")),
    }


def _cleanup_finalize_intermediates(run_dir: Path, state: dict[str, Any]) -> None:
    winner_export = state.get("winner_export", {})
    if isinstance(winner_export, dict):
        for key in ("train_feature_matrix_path", "train_labels_path"):
            value = winner_export.get(key)
            if isinstance(value, str) and value:
                Path(value).expanduser().resolve().unlink(missing_ok=True)
    _finalize_state_path(run_dir).unlink(missing_ok=True)
    cleanup_winner_feature_weights_export(run_dir=run_dir)


def _prepare_supervised_finalize(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    ctx = _context_from_run_dir(run_dir)
    tuning_manifest_path = run_dir / "reports" / "tuning_manifest.json"
    if not tuning_manifest_path.exists():
        raise FileNotFoundError(f"Tuning manifest not found: {tuning_manifest_path}")

    with open(tuning_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if score_percentiles is None:
        score_percentiles = [float(x) for x in manifest["runtime"]["score_percentiles"]]

    task_dir = run_dir / "reports" / "tuning_tasks"
    task_results: list[dict[str, Any]] = []
    missing: list[int] = []
    for task in manifest["tuning"]["tasks"]:
        task_index = int(task["task_index"])
        path = _task_result_path(task_dir, task_index)
        if not path.exists():
            missing.append(task_index)
            continue
        with open(path, "r", encoding="utf-8") as f:
            task_results.append(json.load(f))

    if missing:
        raise RuntimeError(f"Missing tuning task outputs for indices: {missing[:10]}")

    winner = _select_winner(task_results)
    task_spec = _task_spec_from_manifest(manifest)
    features = _load_features_for_tuning_manifest(manifest)
    labels_value = np.load(manifest["data"]["labels_value_path"]).astype(np.int32)
    labels_known = np.load(manifest["data"]["labels_known_path"]).astype(bool)
    rank_values = None
    rank_values_path = manifest["data"].get("rank_values_path")
    if isinstance(rank_values_path, str) and rank_values_path:
        rank_values = np.load(rank_values_path).astype(np.int64)
    with open(manifest["data"]["model_names_path"], "r", encoding="utf-8") as f:
        model_names = [str(x) for x in json.load(f)]
    manifest_items = _load_manifest_items_for_tuning_manifest(manifest)
    manifest_identities = infer_attack_sample_identities(manifest_items)
    identity_by_name = {identity.model_name: identity for identity in manifest_identities}
    missing_identities = [name for name in model_names if name not in identity_by_name]
    if missing_identities:
        preview = ", ".join(missing_identities[:5])
        raise ValueError(
            "Could not align manifest-derived attack identities with stored model names. "
            f"Examples: {preview}"
        )
    all_sample_identities = [identity_by_name[name] for name in model_names]

    train_indices = np.asarray(manifest["data"]["train_indices"], dtype=np.int64)
    train_pool_indices = np.asarray(
        manifest["data"].get("train_pool_indices", manifest["data"]["train_indices"]),
        dtype=np.int64,
    )
    calibration_indices = np.asarray(manifest["data"].get("calibration_indices", []), dtype=np.int64)
    infer_indices = np.asarray(manifest["data"]["infer_indices"], dtype=np.int64)
    x_train = _slice_supervised_features(features, train_indices)
    y_train = labels_value[train_indices]

    model = create(
        str(winner["model_name"]),
        params=dict(winner["params"]),
        random_state=int(manifest["tuning"]["random_state"]),
        task_spec=task_spec,
    )
    winner_backend = model_backend(str(winner["model_name"]))
    if winner_backend == "cnn":
        fit_kwargs: dict[str, Any] = {}
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        if _is_dann_model_name(str(winner["model_name"])):
            fit_indices, fit_kwargs = _resolve_dann_fit_inputs(
                manifest=manifest,
                train_indices=train_indices,
            )
        fit_kwargs.update(
            _resolve_rank_label_fit_kwargs(
                params=dict(winner["params"]),
                rank_values=rank_values,
                fit_indices=fit_indices,
            )
        )
        x_fit = _slice_supervised_features(features, fit_indices)
        model.fit(
            x_fit,
            labels_value[fit_indices],
            n_jobs=int(manifest["tuning"].get("n_jobs", 1)),
            **fit_kwargs,
        )
    else:
        model.fit(x_train, y_train)
    train_outputs = _predict_task_outputs(model, x_train, task_spec=task_spec)
    train_scores = np.asarray(train_outputs.backdoor_scores, dtype=np.float64)
    open_set_unknown_config = _build_open_set_unknown_attack_config(
        train_labels=y_train,
        train_outputs=train_outputs,
        task_spec=task_spec,
    )
    train_open_set_result = _apply_open_set_unknown_attack_rule(
        outputs=train_outputs,
        task_spec=task_spec,
        config=open_set_unknown_config,
    )
    threshold_specs = _build_threshold_specs(
        train_scores=train_scores,
        percentiles=[float(x) for x in score_percentiles],
    )

    model_path = ctx.models_dir / (
        "best_model.pt" 
        if winner_backend in ("cnn", "autoencoder_knn") 
        else "best_model.joblib"
    )
    _save_model(model, model_path)

    # Save fit summary if available (for cnn and autoencoder_knn backends)
    if winner_backend in ("cnn", "autoencoder_knn") and hasattr(model, "fit_summary"):
        fit_summary = model.fit_summary
        if fit_summary:
            fit_summary_path = ctx.models_dir / "fit_summary.json"
            with open(fit_summary_path, "w", encoding="utf-8") as f:
                json.dump(fit_summary, f, indent=2, default=str)

    train_model_names = [model_names[int(i)] for i in train_indices.tolist()]
    train_sample_identities = [all_sample_identities[int(i)] for i in train_indices.tolist()]
    train_task_labels_raw = [int(x) for x in y_train.tolist()]
    train_labels_raw = _project_optional_labels_to_binary(train_task_labels_raw, task_spec=task_spec)
    train_binary_labels_np = task_spec.project_known_labels_to_binary(y_train)
    train_scores_csv = ctx.reports_dir / "train_scores.csv"
    save_score_csv(
        output_path=train_scores_csv,
        model_names=train_model_names,
        labels=train_labels_raw,
        scores=train_scores,
        extra_rows=_multiclass_score_extra_rows(
            task_labels=train_task_labels_raw,
            outputs=train_outputs,
            task_spec=task_spec,
            open_set_result=train_open_set_result,
        ),
    )

    calibration_scores_csv: Path | None = None
    calibration_score_summary: dict[str, Any] | None = None
    calibration_offline_metrics: dict[str, Any] | None = None
    calibration_model_names: list[str] = []
    calibration_labels_raw: list[int] = []
    calibration_task_labels_raw: list[int] = []
    calibration_scores: np.ndarray | None = None
    calibration_outputs: SupervisedPredictionOutputs | None = None
    calibration_open_set_result: dict[str, np.ndarray] | None = None
    calibration_binary_labels_np: np.ndarray | None = None
    if calibration_indices.size > 0:
        x_calibration = _slice_supervised_features(features, calibration_indices)
        y_calibration = labels_value[calibration_indices]
        calibration_outputs = _predict_task_outputs(model, x_calibration, task_spec=task_spec)
        calibration_open_set_result = _apply_open_set_unknown_attack_rule(
            outputs=calibration_outputs,
            task_spec=task_spec,
            config=open_set_unknown_config,
        )
        calibration_scores = np.asarray(calibration_outputs.backdoor_scores, dtype=np.float64)
        calibration_model_names = [model_names[int(i)] for i in calibration_indices.tolist()]
        calibration_task_labels_raw = [int(x) for x in y_calibration.tolist()]
        calibration_labels_raw = _project_optional_labels_to_binary(
            calibration_task_labels_raw,
            task_spec=task_spec,
        )
        calibration_binary_labels_np = task_spec.project_known_labels_to_binary(y_calibration)
        calibration_scores_csv = ctx.reports_dir / "calibration_scores.csv"
        save_score_csv(
            output_path=calibration_scores_csv,
            model_names=calibration_model_names,
            labels=calibration_labels_raw,
            scores=calibration_scores,
            extra_rows=_multiclass_score_extra_rows(
                task_labels=calibration_task_labels_raw,
                outputs=calibration_outputs,
                task_spec=task_spec,
                open_set_result=calibration_open_set_result,
            ),
        )
        calibration_score_summary = summarize_scores(calibration_scores)
        calibration_offline_metrics = compute_offline_metrics(
            calibration_binary_labels_np,
            calibration_scores,
        )

    infer_scores_csv: Path | None = None
    inference_summary: dict[str, Any] | None = None
    threshold_rows: list[dict[str, Any]] = []
    threshold_rows_from_inference: list[dict[str, Any]] = []
    infer_offline_metrics: dict[str, Any] | None = None
    infer_model_names: list[str] = []
    infer_sample_identities: list[AttackSampleIdentity] = []
    infer_labels_raw: list[int | None] = []
    infer_task_labels_raw: list[int | None] = []
    infer_scores: np.ndarray | None = None
    infer_labels_np: np.ndarray | None = None
    infer_task_labels_np: np.ndarray | None = None
    infer_known_task_labels_np: np.ndarray | None = None
    infer_outputs: SupervisedPredictionOutputs | None = None
    infer_open_set_result: dict[str, np.ndarray] | None = None

    if infer_indices.size > 0:
        x_infer = _slice_supervised_features(features, infer_indices)
        infer_outputs = _predict_task_outputs(model, x_infer, task_spec=task_spec)
        infer_open_set_result = _apply_open_set_unknown_attack_rule(
            outputs=infer_outputs,
            task_spec=task_spec,
            config=open_set_unknown_config,
        )
        infer_scores = np.asarray(infer_outputs.backdoor_scores, dtype=np.float64)
        infer_model_names = [model_names[int(i)] for i in infer_indices.tolist()]
        infer_sample_identities = [all_sample_identities[int(i)] for i in infer_indices.tolist()]

        infer_known_mask = labels_known[infer_indices]
        for i in infer_indices.tolist():
            raw = int(labels_value[int(i)])
            infer_task_labels_raw.append(None if raw < 0 else raw)
        infer_labels_raw = _project_optional_labels_to_binary(
            infer_task_labels_raw,
            task_spec=task_spec,
        )
        infer_known_task_labels = [int(x) for x in infer_task_labels_raw if x is not None]
        if infer_known_task_labels:
            infer_known_task_labels_np = np.asarray(infer_known_task_labels, dtype=np.int32)

        if bool(np.all(infer_known_mask)):
            infer_task_labels_np = np.asarray([int(x) for x in infer_task_labels_raw], dtype=np.int32)
            infer_labels_np = task_spec.project_known_labels_to_binary(infer_task_labels_np)

        infer_scores_csv = ctx.reports_dir / "inference_scores.csv"
        save_score_csv(
            output_path=infer_scores_csv,
            model_names=infer_model_names,
            labels=infer_labels_raw,
            scores=infer_scores,
            extra_rows=_multiclass_score_extra_rows(
                task_labels=infer_task_labels_raw,
                outputs=infer_outputs,
                task_spec=task_spec,
                open_set_result=infer_open_set_result,
            ),
        )

        # Extract and save evaluation set features (for cnn and autoencoder_knn backends)
        if winner_backend in ("cnn", "autoencoder_knn") and isinstance(x_infer, SupervisedFeatureBundle):
            infer_features = model.extract_features(x_infer)
            infer_features_path = ctx.reports_dir / "inference_features.npy"
            np.save(infer_features_path, infer_features)

        threshold_rows = compute_infer_threshold_rows(
            train_scores=train_scores,
            infer_scores=infer_scores,
            percentiles=[float(x) for x in score_percentiles],
            infer_labels=infer_labels_np,
        )
        threshold_rows_from_inference = compute_infer_threshold_rows_from_inference(
            infer_scores=infer_scores,
            percentiles=[float(x) for x in score_percentiles],
            infer_labels=infer_labels_np,
        )
        infer_offline_metrics = compute_offline_metrics(
            infer_labels_np,
            infer_scores,
        )
        inference_summary = summarize_scores(infer_scores)

    threshold_selection_cfg = manifest.get("threshold_selection", {})
    selected_threshold_summary: dict[str, Any] | None = None
    selected_threshold_path: Path | None = None
    if calibration_indices.size > 0:
        accepted_fpr_values = threshold_selection_cfg.get("accepted_fprs")
        if accepted_fpr_values is None:
            legacy_value = threshold_selection_cfg.get("accepted_fpr")
            accepted_fpr_values = None if legacy_value is None else [legacy_value]
        resolved_accepted_fprs = _resolve_accepted_fprs(accepted_fpr_values)
        if resolved_accepted_fprs is None:
            raise ValueError(
                "Calibration split is present, but tuning manifest is missing "
                "threshold_selection.accepted_fprs"
            )

        selected_threshold_rows: list[dict[str, Any]] = []
        for accepted_fpr_value in resolved_accepted_fprs:
            selected_threshold_metrics = _select_threshold_max_recall_under_fpr(
                labels_true=np.asarray(calibration_binary_labels_np, dtype=np.int32),
                scores=np.asarray(calibration_scores, dtype=np.float64),
                accepted_fpr=float(accepted_fpr_value),
            )
            inference_selected_metrics: dict[str, Any] | None = None
            if infer_scores is not None and infer_labels_np is not None:
                inference_selected_metrics = _evaluate_binary_threshold(
                    labels_true=infer_labels_np,
                    scores=infer_scores,
                    threshold=float(selected_threshold_metrics["threshold"]),
                )

            selected_threshold_rows.append(
                {
                    "accepted_fpr": float(accepted_fpr_value),
                    "threshold": float(selected_threshold_metrics["threshold"]),
                    "calibration_metrics": selected_threshold_metrics,
                    "inference_metrics": inference_selected_metrics,
                }
            )

        selected_threshold_summary = _build_selected_threshold_summary(
            selections=selected_threshold_rows,
        )
        selected_threshold_path = ctx.reports_dir / SELECTED_THRESHOLD_FILENAME
        with open(selected_threshold_path, "w", encoding="utf-8") as f:
            json.dump(json_ready(selected_threshold_summary), f, indent=2)

    selected_threshold_specs = _build_selected_threshold_specs(selected_threshold_summary)
    train_offline_metrics = compute_offline_metrics(
        train_binary_labels_np,
        train_scores,
    )
    open_set_assessment = None
    if open_set_unknown_config is not None:
        open_set_assessment = {
            "config": open_set_unknown_config,
            "train": _summarize_open_set_partition(
                labels_true=_open_set_true_labels(
                    np.asarray(y_train, dtype=np.int32),
                    task_spec=task_spec,
                    config=open_set_unknown_config,
                ),
                open_set_result=train_open_set_result,
                task_spec=task_spec,
                config=open_set_unknown_config,
            ),
            "calibration": (
                _summarize_open_set_partition(
                    labels_true=_open_set_true_labels(
                        np.asarray(labels_value[calibration_indices], dtype=np.int32),
                        task_spec=task_spec,
                        config=open_set_unknown_config,
                    ),
                    open_set_result=calibration_open_set_result,
                    task_spec=task_spec,
                    config=open_set_unknown_config,
                )
                if calibration_open_set_result is not None
                else None
            ),
            "inference": (
                _summarize_open_set_partition(
                    labels_true=(
                        _open_set_true_labels(
                            np.asarray(infer_task_labels_np, dtype=np.int32),
                            task_spec=task_spec,
                            config=open_set_unknown_config,
                        )
                        if infer_task_labels_np is not None
                        else None
                    ),
                    open_set_result=infer_open_set_result,
                    task_spec=task_spec,
                    config=open_set_unknown_config,
                )
                if infer_open_set_result is not None
                else None
            ),
        }
    multiclass_assessment = None
    if not task_spec.is_binary:
        multiclass_assessment = {
            "train": _summarize_multiclass_partition(
                labels_true=np.asarray(y_train, dtype=np.int32),
                predicted_labels=train_outputs.predicted_labels,
                probabilities=train_outputs.probabilities,
                task_spec=task_spec,
            ),
            "calibration": (
                _summarize_multiclass_partition(
                    labels_true=np.asarray(labels_value[calibration_indices], dtype=np.int32),
                    predicted_labels=np.asarray(calibration_outputs.predicted_labels, dtype=np.int32),
                    probabilities=calibration_outputs.probabilities,
                    task_spec=task_spec,
                )
                if calibration_outputs is not None
                else None
            ),
            "inference": (
                _summarize_multiclass_partition(
                    labels_true=(
                        np.asarray(infer_task_labels_np, dtype=np.int32)
                        if infer_task_labels_np is not None
                        else None
                    ),
                    predicted_labels=(
                        np.asarray(infer_outputs.predicted_labels, dtype=np.int32)
                        if infer_outputs is not None
                        else np.asarray([], dtype=np.int32)
                    ),
                    probabilities=(infer_outputs.probabilities if infer_outputs is not None else None),
                    task_spec=task_spec,
                )
                if infer_outputs is not None
                else None
            ),
        }
    attack_analysis = {
        "train": _summarize_attack_groups(
            sample_identities=train_sample_identities,
            labels=train_labels_raw,
            scores=train_scores,
            threshold_specs=threshold_specs,
            selected_threshold_specs=selected_threshold_specs,
        ),
        "inference": (
            _summarize_attack_groups(
                sample_identities=infer_sample_identities,
                labels=infer_labels_raw,
                scores=np.asarray(infer_scores, dtype=np.float64),
                threshold_specs=threshold_specs,
                selected_threshold_specs=selected_threshold_specs,
            )
            if infer_scores is not None
            else None
        ),
    }

    export_train_features_path: Path | None = None
    export_train_labels_path: Path | None = None
    winner_feature_weights_mode = _winner_feature_weights_mode(
        skip_feature_importance=bool(skip_feature_importance),
        winner_model_name=str(winner["model_name"]),
        task_spec=task_spec,
    )
    report_warnings = [str(x) for x in manifest.get("warnings", [])]
    if winner_feature_weights_mode == "unsupported_for_task_mode":
        report_warnings.append(
            "Skipped winner feature-importance export because multiclass supervised task mode does not "
            "yet support feature-importance export"
        )

    report = {
        "task": task_spec.to_dict(),
        "data_info": {
            "mode": manifest["mode"],
            "n_samples": int(_supervised_feature_row_count(features)),
            "n_train": int(train_indices.size),
            "n_training_pool": int(train_pool_indices.size),
            "n_calibration": int(calibration_indices.size),
            "n_inference": int(infer_indices.size),
            "split": manifest["data"].get("split"),
            "calibration_split": manifest["data"].get("calibration_split"),
            "n_train_clean": int(np.sum(train_binary_labels_np == 0)),
            "n_train_backdoored": int(np.sum(train_binary_labels_np == 1)),
            "n_calibration_clean": (
                int(np.sum(calibration_binary_labels_np == 0))
                if calibration_indices.size > 0
                else 0
            ),
            "n_calibration_backdoored": (
                int(np.sum(calibration_binary_labels_np == 1))
                if calibration_indices.size > 0
                else 0
            ),
            "n_inference_clean": (
                int(sum(1 for label in infer_labels_raw if label == 0))
                if infer_indices.size > 0
                else 0
            ),
            "n_inference_backdoored": (
                int(sum(1 for label in infer_labels_raw if label == 1))
                if infer_indices.size > 0
                else 0
            ),
            "train_class_counts": _named_label_count_rows(np.asarray(y_train, dtype=np.int32), task_spec=task_spec),
            "calibration_class_counts": (
                _named_label_count_rows(
                    np.asarray(labels_value[calibration_indices], dtype=np.int32),
                    task_spec=task_spec,
                )
                if calibration_indices.size > 0
                else []
            ),
            "inference_class_counts": (
                _named_label_count_rows(np.asarray(infer_known_task_labels_np, dtype=np.int32), task_spec=task_spec)
                if infer_known_task_labels_np is not None
                else []
            ),
            "n_inference_unknown_label": (
                int(np.sum(labels_value[infer_indices] < 0))
                if infer_indices.size > 0
                else 0
            ),
        },
        "representation": {
            "extractor": manifest["extractor"]["name"],
            "extractor_params": manifest["extractor"]["params"],
            "extractor_metadata": _compact_extractor_metadata(manifest["extractor"]["metadata"]),
            "feature_path": manifest["data"]["feature_path"],
        },
        "tuning": {
            "metric": manifest["tuning"].get("metric", task_spec.selection_metric_name),
            "model_name": manifest["tuning"]["model_name"],
            "model_names": manifest["tuning"].get("model_names", [manifest["tuning"]["model_name"]]),
            "cnn_hyperparams": manifest["tuning"].get("cnn_hyperparams"),
            "domain_adaptation": manifest.get("domain_adaptation"),
            "execution_mode": manifest["tuning"].get("execution_mode", "cross_validation"),
            "class_weight_loss": bool(manifest["tuning"].get("class_weight_loss", False)),
            "rank_label_weight_loss": bool(manifest["tuning"].get("rank_label_weight_loss", False)),
            "cv_random_states": manifest["tuning"].get("cv_random_states", [manifest["tuning"]["random_state"]]),
            "cv_folds_resolved": manifest["tuning"]["cv_folds_resolved"],
            "cv_stratification": manifest["tuning"].get("cv_stratification"),
            "estimated_total_fits": manifest["tuning"].get("estimated_total_fits"),
            "executor": manifest["tuning"]["executor"],
            "tasks_total": len(manifest["tuning"]["tasks"]),
            "candidates": task_results,
            "winner": winner,
        },
        "fit_assessment": {
            "score_definition": (
                "positive_class_score"
                if task_spec.binary_projection == BINARY_PROJECTION_POSITIVE_CLASS_SCORE
                else "backdoor_score"
            ),
            "binary_projection": str(task_spec.binary_projection),
            "train_score_summary": summarize_scores(train_scores),
            "train_offline_metrics": train_offline_metrics,
            "calibration_score_summary": calibration_score_summary,
            "calibration_offline_metrics": calibration_offline_metrics,
            "inference_score_summary": inference_summary,
            "threshold_evaluation": threshold_rows,
            "threshold_evaluation_from_inference": threshold_rows_from_inference,
            "offline_metrics": infer_offline_metrics,
        },
        "multiclass_assessment": multiclass_assessment,
        "open_set_assessment": open_set_assessment,
        "threshold_selection": selected_threshold_summary,
        "attack_analysis": attack_analysis,
        "warnings": report_warnings,
    }

    report_path = ctx.reports_dir / "supervised_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    from ..experiments.supervised_architecture_breakdown import (
        build_supervised_results_summary,
        write_supervised_results_summary_outputs,
    )

    results_summary = build_supervised_results_summary(run_dir=run_dir)
    results_summary_outputs = write_supervised_results_summary_outputs(
        run_dir=run_dir,
        summary=results_summary,
        update_report=True,
        update_artifact_index=False,
        remove_legacy_outputs=True,
    )
    results_summary_markdown_path = results_summary_outputs["results_summary_md"]

    if winner_feature_weights_mode == "pending":
        export_train_features_path = ctx.features_dir / FINALIZE_EXPORT_TRAIN_FEATURES_FILENAME
        export_train_labels_path = ctx.features_dir / FINALIZE_EXPORT_TRAIN_LABELS_FILENAME
        np.save(export_train_features_path, np.asarray(_tabular_array_or_raise(x_train), dtype=np.float32))
        np.save(export_train_labels_path, np.asarray(y_train, dtype=np.int32))

    run_config = {
        "pipeline": PIPELINE_NAME,
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_json": manifest["manifest_json"],
        "dataset_root": manifest["dataset_root"],
        "mode": manifest["mode"],
        "tuning_executor": manifest["tuning"]["executor"],
        "task": task_spec.to_dict(),
        "model_name": manifest["tuning"]["model_name"],
        "model_names": manifest["tuning"].get("model_names", [manifest["tuning"]["model_name"]]),
        "cnn_hyperparams": manifest["tuning"].get("cnn_hyperparams"),
        "domain_adaptation": manifest.get("domain_adaptation"),
        "execution_mode": manifest["tuning"].get("execution_mode", "cross_validation"),
        "train_split_percent": int(manifest["tuning"].get("train_split_percent", 100)),
        "calibration_split_percent": manifest["tuning"].get("calibration_split_percent"),
        "accepted_fprs": threshold_selection_cfg.get("accepted_fprs"),
        "accepted_fpr": threshold_selection_cfg.get("accepted_fpr"),
        "split_by_folder": bool(manifest["tuning"].get("split_by_folder", False)),
        "class_weight_loss": bool(manifest["tuning"].get("class_weight_loss", False)),
        "rank_label_weight_loss": bool(manifest["tuning"].get("rank_label_weight_loss", False)),
        "data_split": manifest["data"].get("split"),
        "calibration_split": manifest["data"].get("calibration_split"),
        "cv_random_states": manifest["tuning"].get("cv_random_states", [manifest["tuning"]["random_state"]]),
        "cv_stratification": manifest["tuning"].get("cv_stratification"),
        "score_percentiles": [float(x) for x in score_percentiles],
        "winner": winner,
        "threshold_selection": selected_threshold_summary,
        "open_set_unknown_attack": open_set_unknown_config,
        "skip_feature_importance": bool(skip_feature_importance),
        "winner_feature_weights_mode": winner_feature_weights_mode,
        "warnings": report_warnings,
    }
    artifacts = {
        "best_model": str(model_path),
        "train_scores_csv": str(train_scores_csv),
        "calibration_scores_csv": str(calibration_scores_csv) if calibration_scores_csv is not None else None,
        "inference_scores_csv": str(infer_scores_csv) if infer_scores_csv is not None else None,
        "selected_threshold": str(selected_threshold_path) if selected_threshold_path is not None else None,
        "report": str(report_path),
        "results_summary_md": str(results_summary_markdown_path),
        "tuning_manifest": str(tuning_manifest_path),
        "tuning_tasks_dir": str(task_dir),
    }
    state_path = _write_finalize_state(
        run_dir=run_dir,
        run_config=run_config,
        artifacts=artifacts,
        winner_export_train_features=export_train_features_path,
        winner_export_train_labels=export_train_labels_path,
        random_state=int(manifest["tuning"]["random_state"]),
        skip_feature_importance=bool(skip_feature_importance),
    )

    return {
        "run_dir": str(run_dir),
        "report": str(report_path),
        "train_scores_csv": str(train_scores_csv),
        "calibration_scores_csv": str(calibration_scores_csv) if calibration_scores_csv is not None else None,
        "inference_scores_csv": str(infer_scores_csv) if infer_scores_csv is not None else None,
        "best_model": str(model_path),
        "results_summary_md": str(results_summary_markdown_path),
        "train_features": x_train,
        "train_labels": y_train,
        "random_state": int(manifest["tuning"]["random_state"]),
        "tuning_manifest_path": tuning_manifest_path,
        "finalize_state_path": state_path,
        "winner_export_train_features": export_train_features_path,
        "winner_export_train_labels": export_train_labels_path,
        "winner_feature_weights_mode": winner_feature_weights_mode,
    }


def _complete_supervised_finalize(
    *,
    run_dir: Path,
    winner_exports: dict[str, Path] | None,
) -> dict[str, Any]:
    state = _load_finalize_state(run_dir)
    ctx = _context_from_run_dir(run_dir)

    base_artifacts = state.get("artifacts", {})
    if not isinstance(base_artifacts, dict):
        raise ValueError("Finalize state is missing base artifacts")
    for key, value in base_artifacts.items():
        if value:
            ctx.add_artifact(str(key), Path(str(value)))

    if winner_exports is not None:
        ctx.add_artifact("winner_feature_weights_coefficients_csv", winner_exports["coefficient_csv"])
        ctx.add_artifact("winner_feature_weights_by_metric_csv", winner_exports["metric_csv"])
        ctx.add_artifact("winner_feature_weights_by_block_csv", winner_exports["block_csv"])
        ctx.add_artifact("winner_feature_weights_metadata_json", winner_exports["metadata_json"])

    run_config = state.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Finalize state is missing run_config")
    ctx.finalize(run_config)
    _cleanup_finalize_intermediates(run_dir, state)

    return {
        "run_dir": str(run_dir),
        "report": str(base_artifacts.get("report")),
        "train_scores_csv": str(base_artifacts.get("train_scores_csv")),
        "inference_scores_csv": (
            str(base_artifacts.get("inference_scores_csv"))
            if base_artifacts.get("inference_scores_csv")
            else None
        ),
        "best_model": str(base_artifacts.get("best_model")),
    }


def _prepare_supervised_finalize_distributed(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
    finalize_export_shards: int,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    prepared = _prepare_supervised_finalize(
        run_dir=run_dir,
        score_percentiles=score_percentiles,
        skip_feature_importance=skip_feature_importance,
    )
    if prepared["winner_feature_weights_mode"] != "pending":
        finalized = _complete_supervised_finalize(
            run_dir=run_dir,
            winner_exports=None,
        )
        return {
            **finalized,
            "winner_feature_weights_mode": prepared["winner_feature_weights_mode"],
            "winner_feature_weights_tasks": 0,
        }
    if prepared["winner_export_train_features"] is None or prepared["winner_export_train_labels"] is None:
        raise RuntimeError("Finalize export inputs were not materialized for distributed feature importance")
    export_prepared = prepare_winner_feature_weights_export(
        run_dir=run_dir,
        report_path=Path(prepared["report"]),
        manifest_path=Path(prepared["tuning_manifest_path"]),
        artifact_index_path=run_dir / "artifact_index.json",
        train_feature_matrix_path=Path(prepared["winner_export_train_features"]),
        train_labels_path=Path(prepared["winner_export_train_labels"]),
        random_state=int(prepared["random_state"]),
        n_tasks=int(finalize_export_shards),
    )
    return {
        "run_dir": str(run_dir),
        "report": prepared["report"],
        "train_scores_csv": prepared["train_scores_csv"],
        "inference_scores_csv": prepared["inference_scores_csv"],
        "best_model": prepared["best_model"],
        "winner_feature_weights_mode": export_prepared["mode"],
        "winner_feature_weights_manifest": str(export_prepared["manifest_path"]),
        "winner_feature_weights_tasks": int(export_prepared["n_tasks"]),
    }


def _run_supervised_finalize_worker(
    *,
    run_dir: Path,
    task_index: int,
    n_jobs: int | None,
) -> dict[str, Any]:
    finalized_mode = _already_finalized_winner_feature_weights_mode(run_dir)
    if finalized_mode is not None:
        return {
            "status": "skipped",
            "reason": f"winner feature export already finalized for mode={finalized_mode!r}",
        }
    return run_winner_feature_weights_export_worker(
        run_dir=run_dir,
        task_index=task_index,
        n_jobs=1 if n_jobs is None else int(n_jobs),
    )


def _merge_supervised_finalize(
    *,
    run_dir: Path,
) -> dict[str, Any]:
    finalized_mode = _already_finalized_winner_feature_weights_mode(run_dir)
    if finalized_mode is not None:
        return _completed_supervised_result(run_dir)
    winner_exports = merge_winner_feature_weights_export(
        run_dir=run_dir,
        artifact_index_path=run_dir / "artifact_index.json",
    )
    return _complete_supervised_finalize(
        run_dir=run_dir,
        winner_exports=winner_exports,
    )


def _finalize_supervised_run(
    *,
    run_dir: Path,
    score_percentiles: list[float] | None,
    n_jobs: int,
    skip_feature_importance: bool,
) -> dict[str, Any]:
    prepared = _prepare_supervised_finalize(
        run_dir=run_dir,
        score_percentiles=score_percentiles,
        skip_feature_importance=skip_feature_importance,
    )
    if prepared["winner_feature_weights_mode"] != "pending":
        finalized = _complete_supervised_finalize(
            run_dir=run_dir,
            winner_exports=None,
        )
        return {
            **finalized,
            "winner_feature_weights_mode": prepared["winner_feature_weights_mode"],
        }
    winner_exports = export_winner_feature_weights(
        run_dir=run_dir,
        report_path=Path(prepared["report"]),
        manifest_path=Path(prepared["tuning_manifest_path"]),
        artifact_index_path=run_dir / "artifact_index.json",
        train_features=np.asarray(prepared["train_features"]),
        train_labels=np.asarray(prepared["train_labels"]),
        random_state=int(prepared["random_state"]),
        n_jobs=int(n_jobs),
    )
    return _complete_supervised_finalize(
        run_dir=run_dir,
        winner_exports=winner_exports,
    )


def run_supervised_pipeline(
    *,
    manifest_json: Path | None,
    dataset_root: Path,
    output_root: Path,
    run_id: str | None,
    model_name: str,
    spectral_features: list[str] | None,
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    stream_block_size: int,
    dtype_name: str,
    cv_folds: int,
    random_state: int,
    train_split_percent: int,
    calibration_split_percent: int | None,
    accepted_fpr: float | list[float] | tuple[float, ...] | None,
    split_by_folder: bool,
    cv_random_states: list[int] | None,
    n_jobs: int,
    score_percentiles: list[float] | None,
    feature_file: Path | None,
    tuning_executor: str,
    slurm_partition: str,
    slurm_max_concurrent: str,
    slurm_cpus_per_task: str,
    finalize_export_shards: int,
    stage: str,
    run_dir: Path | None,
    task_index: int | None,
    task_mode: str = SUPERVISED_TASK_MODE_BINARY,
    multiclass_attack_names: list[str] | None = None,
    cnn_hyperparams: Path | None = None,
    autoencoder_hyperparams: Path | None = None,
    dann_source_rank: int = DANN_DEFAULT_SOURCE_RANK,
    dann_target_adaptation_percent: int = DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
    dann_lambda_max: float = DANN_DEFAULT_LAMBDA_MAX,
    dann_lambda_gamma: float = DANN_DEFAULT_LAMBDA_GAMMA,
    dann_lr_alpha: float = DANN_DEFAULT_LR_ALPHA,
    dann_lr_beta: float = DANN_DEFAULT_LR_BETA,
    class_weight_loss: bool = False,
    rank_label_weight_loss: bool = False,
    skip_feature_importance: bool = False,
    selection_metric: str | None = None,
) -> dict[str, Any]:
    if stage in {"all", "prepare"} and manifest_json is None:
        raise ValueError("--manifest-json is required for stage=all and stage=prepare")
    if stage in {"all", "prepare"} and feature_file is None:
        raise ValueError(
            "--feature-file is required for stage=all and stage=prepare. Pass a feature run name, "
            "feature output directory, or spectral_features.npy path."
        )
    if stage in {"all", "prepare"} and not spectral_features:
        raise ValueError(
            "--features is required for stage=all and stage=prepare. Specify the feature groups to "
            "select from the extracted feature bundle."
        )
    if stage in {"worker", "finalize", "finalize_prepare", "finalize_worker", "finalize_merge"} and run_dir is None:
        raise ValueError("--run-dir is required for the selected supervised stage")
    if stage in {"worker", "finalize_worker"} and task_index is None:
        raise ValueError("--task-index is required for the selected supervised worker stage")
    resolved_accepted_fprs = _resolve_accepted_fprs(accepted_fpr)
    if (calibration_split_percent is None) != (resolved_accepted_fprs is None):
        raise ValueError("--calibration-split and --accepted-fpr must either both be set or both be omitted")

    if stage == "prepare":
        return _prepare_supervised_run(
            manifest_json=Path(manifest_json) if manifest_json is not None else Path(""),
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=run_id,
            model_name=model_name,
            spectral_features=spectral_features,
            spectral_sv_top_k=spectral_sv_top_k,
            spectral_moment_source=spectral_moment_source,
            spectral_qv_sum_mode=spectral_qv_sum_mode,
            spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
            stream_block_size=stream_block_size,
            dtype_name=dtype_name,
            cv_folds=cv_folds,
            random_state=random_state,
            train_split_percent=train_split_percent,
            calibration_split_percent=calibration_split_percent,
            accepted_fpr=resolved_accepted_fprs,
            split_by_folder=split_by_folder,
            cv_random_states=cv_random_states,
            n_jobs=n_jobs,
            score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
            feature_file=feature_file,
            tuning_executor=tuning_executor,
            slurm_partition=slurm_partition,
            slurm_max_concurrent=slurm_max_concurrent,
            slurm_cpus_per_task=slurm_cpus_per_task,
            task_mode=task_mode,
            multiclass_attack_names=multiclass_attack_names,
            cnn_hyperparams=cnn_hyperparams,
            autoencoder_hyperparams=autoencoder_hyperparams,
            dann_source_rank=int(dann_source_rank),
            dann_target_adaptation_percent=int(dann_target_adaptation_percent),
            dann_lambda_max=float(dann_lambda_max),
            dann_lambda_gamma=float(dann_lambda_gamma),
            dann_lr_alpha=float(dann_lr_alpha),
            dann_lr_beta=float(dann_lr_beta),
            class_weight_loss=bool(class_weight_loss),
            rank_label_weight_loss=bool(rank_label_weight_loss),
            selection_metric=selection_metric,
        )

    if stage == "worker":
        return _run_supervised_worker(
            run_dir=Path(run_dir),
            task_index=int(task_index) if task_index is not None else -1,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
        )

    if stage == "finalize":
        return _finalize_supervised_run(
            run_dir=Path(run_dir),
            score_percentiles=score_percentiles,
            n_jobs=int(n_jobs),
            skip_feature_importance=bool(skip_feature_importance),
        )

    if stage == "finalize_prepare":
        return _prepare_supervised_finalize_distributed(
            run_dir=Path(run_dir),
            score_percentiles=score_percentiles,
            finalize_export_shards=int(finalize_export_shards),
            skip_feature_importance=bool(skip_feature_importance),
        )

    if stage == "finalize_worker":
        return _run_supervised_finalize_worker(
            run_dir=Path(run_dir),
            task_index=int(task_index) if task_index is not None else -1,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
        )

    if stage == "finalize_merge":
        return _merge_supervised_finalize(
            run_dir=Path(run_dir),
        )

    # stage == "all"
    prepared = _prepare_supervised_run(
        manifest_json=Path(manifest_json) if manifest_json is not None else Path(""),
        dataset_root=dataset_root,
        output_root=output_root,
        run_id=run_id,
        model_name=model_name,
        spectral_features=spectral_features,
        spectral_sv_top_k=spectral_sv_top_k,
        spectral_moment_source=spectral_moment_source,
        spectral_qv_sum_mode=spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=spectral_entrywise_delta_mode,
        stream_block_size=stream_block_size,
        dtype_name=dtype_name,
        cv_folds=cv_folds,
        random_state=random_state,
        train_split_percent=train_split_percent,
        calibration_split_percent=calibration_split_percent,
        accepted_fpr=resolved_accepted_fprs,
        split_by_folder=split_by_folder,
        cv_random_states=cv_random_states,
        n_jobs=n_jobs,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
        feature_file=feature_file,
        tuning_executor=tuning_executor,
        slurm_partition=slurm_partition,
        slurm_max_concurrent=slurm_max_concurrent,
        slurm_cpus_per_task=slurm_cpus_per_task,
        task_mode=task_mode,
        multiclass_attack_names=multiclass_attack_names,
        cnn_hyperparams=cnn_hyperparams,
        autoencoder_hyperparams=autoencoder_hyperparams,
        dann_source_rank=int(dann_source_rank),
        dann_target_adaptation_percent=int(dann_target_adaptation_percent),
        dann_lambda_max=float(dann_lambda_max),
        dann_lambda_gamma=float(dann_lambda_gamma),
        dann_lr_alpha=float(dann_lr_alpha),
        dann_lr_beta=float(dann_lr_beta),
        class_weight_loss=bool(class_weight_loss),
        rank_label_weight_loss=bool(rank_label_weight_loss),
        selection_metric=selection_metric,
    )
    resolved_run_dir = Path(prepared["run_dir"])

    if tuning_executor == "slurm_array":
        n_tasks = int(prepared["n_tasks"])
        max_concurrent = int(prepared["slurm_max_concurrent"])
        return {
            **prepared,
            "next_steps": build_slurm_array_next_steps(
                run_dir=resolved_run_dir,
                n_tasks=n_tasks,
                max_concurrent=max_concurrent,
                skip_feature_importance=bool(skip_feature_importance),
            ),
        }

    for idx in range(int(prepared["n_tasks"])):
        _run_supervised_worker(
            run_dir=resolved_run_dir,
            task_index=idx,
            n_jobs=n_jobs,
        )

    finalized = _finalize_supervised_run(
        run_dir=resolved_run_dir,
        score_percentiles=score_percentiles or [90.0, 95.0, 99.0],
        n_jobs=n_jobs,
        skip_feature_importance=bool(skip_feature_importance),
    )
    return {
        **finalized,
        "tuning_manifest": prepared["tuning_manifest"],
        "n_tasks": prepared["n_tasks"],
    }

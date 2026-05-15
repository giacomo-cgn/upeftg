from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .interfaces import (
    ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND,
    BINARY_PROJECTION_ONE_MINUS_CLEAN_PROBABILITY,
    BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    SUPERVISED_TASK_MODE_BINARY,
    SupervisedTaskSpec,
    SupervisedFeatureBundle,
)

try:  # pragma: no cover - exercised in environments that install torch
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - soft dependency
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None


CNN_MAX_EPOCHS = 40
CNN_BATCH_SIZE = 16
CNN_PATIENCE = 5
_DEPTH_LABEL_PATTERN = re.compile(r"^(encoder|decoder)\.layer(\d+)$")
_SUPPORTED_ATTENTION_KINDS = ("self", "cross")
TorchModuleBase = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or F is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ModuleNotFoundError(
            "cnn_1d requires the optional 'torch' dependency, but torch is not installed"
        )


def _default_binary_task_spec() -> SupervisedTaskSpec:
    class_names = ("clean", "backdoored")
    return SupervisedTaskSpec(
        task_mode=SUPERVISED_TASK_MODE_BINARY,
        class_names=class_names,
        class_to_index={name: idx for idx, name in enumerate(class_names)},
        binary_projection=BINARY_PROJECTION_POSITIVE_CLASS_SCORE,
    )


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


@dataclass(frozen=True)
class CNNFeatureTensors:
    inputs: np.ndarray
    layer_mask: np.ndarray


@dataclass(frozen=True)
class CNNLayerVectorConfig:
    conv_channels: int = 64
    num_conv_layers: int = 3
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    dropout: float = 0.1
    use_residual: bool = True
    normalization: str = "layernorm"
    pooling: str = "mean_max"
    include_total_layer_count: bool = True
    depth_feature_mode: str = "both"


@dataclass(frozen=True)
class CNNNormalizationStats:
    channel_mean: np.ndarray
    channel_std: np.ndarray


@dataclass(frozen=True)
class CNNChannelLayout:
    slot_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    value_start: int
    value_end: int
    value_mask_start: int
    value_mask_end: int
    slot_presence_start: int
    slot_presence_end: int
    self_attention_index: int
    cross_attention_index: int
    is_encoder_index: int
    is_decoder_index: int
    layer_index_index: int | None
    normalized_arch_index_index: int | None
    normalized_sequence_index_index: int | None
    total_layers_index: int | None
    continuous_indices: tuple[int, ...]

    @property
    def input_dim(self) -> int:
        return int(
            max(
                self.value_end,
                self.value_mask_end,
                self.slot_presence_end,
                self.cross_attention_index + 1,
                self.is_decoder_index + 1,
                (self.layer_index_index + 1) if self.layer_index_index is not None else 0,
                (
                    self.normalized_arch_index_index + 1
                    if self.normalized_arch_index_index is not None
                    else 0
                ),
                (
                    self.normalized_sequence_index_index + 1
                    if self.normalized_sequence_index_index is not None
                    else 0
                ),
                (self.total_layers_index + 1) if self.total_layers_index is not None else 0,
            )
        )


@dataclass(frozen=True)
class CNNLayerVectorBatch:
    sequences: tuple[np.ndarray, ...]
    continuous_valid_masks: tuple[np.ndarray, ...]
    layer_names: tuple[tuple[str, ...], ...]
    channel_layout: CNNChannelLayout
    normalization_stats: CNNNormalizationStats


def compute_balanced_class_loss_config(
    labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    labels_np = np.asarray(labels).reshape(-1)
    if sample_mask is None:
        selected = labels_np
    else:
        mask_np = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if mask_np.shape[0] != labels_np.shape[0]:
            raise ValueError("Class-weight loss mask must have the same length as labels")
        selected = labels_np[mask_np]
    selected_int = np.asarray(selected, dtype=np.int64).reshape(-1)
    if selected_int.size == 0:
        raise ValueError("Class-weight loss requires at least one labeled row")

    class_counts = np.asarray(
        [int(np.sum(selected_int == int(class_idx))) for class_idx in range(task_spec.n_classes)],
        dtype=np.int64,
    )
    present_mask = class_counts > 0
    present_count = int(np.sum(present_mask))
    if present_count == 0:
        raise ValueError("Class-weight loss requires at least one observed class")

    class_weights = np.ones(task_spec.n_classes, dtype=np.float32)
    total_present = int(np.sum(class_counts[present_mask]))
    class_weights[present_mask] = (
        float(total_present) / (float(present_count) * class_counts[present_mask].astype(np.float32))
    )

    binary_pos_weight: float | None = None
    if task_spec.is_binary:
        negative_count = int(class_counts[0]) if class_counts.shape[0] > 0 else 0
        positive_count = int(class_counts[1]) if class_counts.shape[0] > 1 else 0
        binary_pos_weight = (
            float(negative_count) / float(positive_count)
            if negative_count > 0 and positive_count > 0
            else 1.0
        )

    return {
        "class_counts": [int(x) for x in class_counts.tolist()],
        "class_weights": [float(x) for x in class_weights.tolist()],
        "binary_pos_weight": (None if binary_pos_weight is None else float(binary_pos_weight)),
    }


def compute_balanced_rank_label_loss_config(
    labels: np.ndarray,
    rank_labels: np.ndarray,
    *,
    task_spec: SupervisedTaskSpec,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels_np = np.asarray(labels).reshape(-1)
    ranks_np = np.asarray(rank_labels).reshape(-1)
    if ranks_np.shape[0] != labels_np.shape[0]:
        raise ValueError("Rank-label loss ranks must have the same length as labels")

    if sample_mask is None:
        mask_np = np.ones(labels_np.shape[0], dtype=bool)
    else:
        mask_np = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if mask_np.shape[0] != labels_np.shape[0]:
            raise ValueError("Rank-label loss mask must have the same length as labels")

    selected_labels = np.asarray(labels_np[mask_np], dtype=np.int64).reshape(-1)
    selected_ranks = np.asarray(ranks_np[mask_np], dtype=np.int64).reshape(-1)
    if selected_labels.size == 0:
        raise ValueError("Rank-label loss requires at least one labeled row")
    if bool(np.any(selected_labels < 0)) or bool(np.any(selected_labels >= int(task_spec.n_classes))):
        raise ValueError("Rank-label loss requires labels to be valid supervised class indices")

    bucket_counts: dict[tuple[int, int], int] = {}
    for rank, label in zip(selected_ranks.tolist(), selected_labels.tolist()):
        key = (int(rank), int(label))
        bucket_counts[key] = int(bucket_counts.get(key, 0) + 1)
    if not bucket_counts:
        raise ValueError("Rank-label loss requires at least one observed rank-label bucket")

    total_present = int(selected_labels.size)
    bucket_count = int(len(bucket_counts))
    bucket_weights = {
        key: float(total_present) / (float(bucket_count) * float(count))
        for key, count in bucket_counts.items()
    }

    sample_weights = np.zeros(labels_np.shape[0], dtype=np.float32)
    selected_indices = np.flatnonzero(mask_np)
    for idx, rank, label in zip(selected_indices.tolist(), selected_ranks.tolist(), selected_labels.tolist()):
        sample_weights[int(idx)] = np.float32(bucket_weights[(int(rank), int(label))])

    class_counts = [
        int(np.sum(selected_labels == int(class_idx)))
        for class_idx in range(int(task_spec.n_classes))
    ]
    rank_counts = [
        {
            "rank": int(rank),
            "count": int(np.sum(selected_ranks == int(rank))),
        }
        for rank in sorted({int(x) for x in selected_ranks.tolist()})
    ]
    rank_label_counts = [
        {
            "rank": int(rank),
            "label": int(label),
            "class_name": (
                str(task_spec.class_names[int(label)])
                if 0 <= int(label) < len(task_spec.class_names)
                else str(label)
            ),
            "count": int(bucket_counts[(int(rank), int(label))]),
            "weight": float(bucket_weights[(int(rank), int(label))]),
        }
        for rank, label in sorted(bucket_counts)
    ]
    selected_weights = sample_weights[mask_np]
    config = {
        "class_counts": class_counts,
        "rank_counts": rank_counts,
        "rank_label_counts": rank_label_counts,
        "sample_weight_min": float(np.min(selected_weights)),
        "sample_weight_max": float(np.max(selected_weights)),
        "sample_weight_mean": float(np.mean(selected_weights)),
    }
    return sample_weights, config


def _resolve_bundle_metadata(
    bundle: SupervisedFeatureBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    if bundle.representation_kind != ARCHITECTURE_INDEPENDENT_LAYER_SEQUENCE_KIND:
        raise ValueError(
            "cnn_1d requires architecture_independent_layer_sequence_aggregation input, "
            f"got {bundle.representation_kind!r}"
        )
    if bundle.group_mask is None or bundle.value_mask is None:
        raise ValueError("cnn_1d requires group_mask and value_mask companions")

    values = np.asarray(bundle.values, dtype=np.float32)
    group_mask = np.asarray(bundle.group_mask, dtype=bool)
    value_mask = np.asarray(bundle.value_mask, dtype=bool)
    if values.ndim != 4:
        raise ValueError(f"cnn_1d expects a 4D tensor, got shape={values.shape}")
    if group_mask.shape != values.shape[:2]:
        raise ValueError(
            f"group_mask shape {group_mask.shape} does not match tensor depth shape {values.shape[:2]}"
        )
    if value_mask.shape != values.shape:
        raise ValueError(
            f"value_mask shape {value_mask.shape} does not match tensor shape {values.shape}"
        )

    metadata = bundle.metadata if isinstance(bundle.metadata, dict) else {}
    raw_depth_labels = metadata.get("depth_labels")
    if not isinstance(raw_depth_labels, list) or len(raw_depth_labels) != int(values.shape[1]):
        raise ValueError(
            "cnn_1d requires metadata.depth_labels aligned to the canonical depth axis"
        )
    raw_slot_names = metadata.get("slot_names")
    if not isinstance(raw_slot_names, list) or len(raw_slot_names) != int(values.shape[2]):
        raise ValueError(
            "cnn_1d requires metadata.slot_names aligned to the canonical slot axis"
        )
    raw_feature_names = metadata.get("emitted_feature_names")
    if not isinstance(raw_feature_names, list) or len(raw_feature_names) != int(values.shape[3]):
        raw_feature_names = [f"feature_{idx}" for idx in range(int(values.shape[3]))]

    return (
        values,
        group_mask,
        value_mask,
        [str(x) for x in raw_depth_labels],
        [str(x) for x in raw_slot_names],
        [str(x) for x in raw_feature_names],
    )


def _parse_depth_label(depth_label: str) -> tuple[str, int]:
    match = _DEPTH_LABEL_PATTERN.match(str(depth_label).strip())
    if match is None:
        raise ValueError(
            "cnn_1d expected canonical depth labels like 'encoder.layer12' or 'decoder.layer7', "
            f"got {depth_label!r}"
        )
    return str(match.group(1)), int(match.group(2))


def _normalized_position(index: int, upper_bound: int) -> float:
    if upper_bound <= 0:
        return 0.0
    return float(index) / float(upper_bound)


def _resolve_retained_slot_indices(
    slot_names: list[str],
    *,
    group_mask: np.ndarray,
    value_mask: np.ndarray,
) -> list[int]:
    retained_indices: list[int] = []
    populated_other_slots: list[str] = []
    valid_depth_mask = np.asarray(group_mask[:, :, None], dtype=bool)

    for slot_idx, slot_name in enumerate(slot_names):
        attention_kind = str(slot_name).partition(".")[0]
        if attention_kind in _SUPPORTED_ATTENTION_KINDS:
            retained_indices.append(int(slot_idx))
            continue
        slot_populated = bool(
            np.logical_and(valid_depth_mask, value_mask[:, :, int(slot_idx), :]).any()
        )
        if slot_populated:
            populated_other_slots.append(str(slot_name))

    if populated_other_slots:
        preview = ", ".join(populated_other_slots[:5])
        raise ValueError(
            "cnn_1d only supports self/cross attention slots, but the layer-sequence bundle "
            f"contains populated unsupported slot(s): {preview}"
        )
    if not retained_indices:
        raise ValueError("cnn_1d requires at least one populated self/cross attention slot")
    return retained_indices


def _build_channel_layout(
    *,
    retained_slot_names: list[str],
    feature_names: list[str],
    include_total_layer_count: bool,
    depth_feature_mode: str,
) -> CNNChannelLayout:
    if depth_feature_mode not in {"absolute", "normalized", "both"}:
        raise ValueError(
            "cnn_1d depth_feature_mode must be one of {'absolute', 'normalized', 'both'}"
        )

    value_width = int(len(retained_slot_names) * len(feature_names))
    cursor = 0
    value_start = cursor
    value_end = value_start + value_width
    cursor = value_end

    value_mask_start = cursor
    value_mask_end = value_mask_start + value_width
    cursor = value_mask_end

    slot_presence_start = cursor
    slot_presence_end = slot_presence_start + int(len(retained_slot_names))
    cursor = slot_presence_end

    self_attention_index = cursor
    cursor += 1
    cross_attention_index = cursor
    cursor += 1
    is_encoder_index = cursor
    cursor += 1
    is_decoder_index = cursor
    cursor += 1

    layer_index_index = None
    if depth_feature_mode in {"absolute", "both"}:
        layer_index_index = cursor
        cursor += 1

    normalized_arch_index_index = None
    normalized_sequence_index_index = None
    if depth_feature_mode in {"normalized", "both"}:
        normalized_arch_index_index = cursor
        cursor += 1
        normalized_sequence_index_index = cursor
        cursor += 1

    total_layers_index = None
    if include_total_layer_count:
        total_layers_index = cursor
        cursor += 1

    continuous_indices = list(range(value_start, value_end))
    if layer_index_index is not None:
        continuous_indices.append(int(layer_index_index))
    if normalized_arch_index_index is not None:
        continuous_indices.append(int(normalized_arch_index_index))
    if normalized_sequence_index_index is not None:
        continuous_indices.append(int(normalized_sequence_index_index))
    if total_layers_index is not None:
        continuous_indices.append(int(total_layers_index))

    return CNNChannelLayout(
        slot_names=tuple(str(x) for x in retained_slot_names),
        feature_names=tuple(str(x) for x in feature_names),
        value_start=int(value_start),
        value_end=int(value_end),
        value_mask_start=int(value_mask_start),
        value_mask_end=int(value_mask_end),
        slot_presence_start=int(slot_presence_start),
        slot_presence_end=int(slot_presence_end),
        self_attention_index=int(self_attention_index),
        cross_attention_index=int(cross_attention_index),
        is_encoder_index=int(is_encoder_index),
        is_decoder_index=int(is_decoder_index),
        layer_index_index=None if layer_index_index is None else int(layer_index_index),
        normalized_arch_index_index=(
            None
            if normalized_arch_index_index is None
            else int(normalized_arch_index_index)
        ),
        normalized_sequence_index_index=(
            None
            if normalized_sequence_index_index is None
            else int(normalized_sequence_index_index)
        ),
        total_layers_index=None if total_layers_index is None else int(total_layers_index),
        continuous_indices=tuple(int(x) for x in continuous_indices),
    )


def _compute_normalization_stats(
    *,
    sequences: tuple[np.ndarray, ...],
    continuous_valid_masks: tuple[np.ndarray, ...],
    input_dim: int,
) -> CNNNormalizationStats:
    means = np.zeros(int(input_dim), dtype=np.float32)
    std = np.ones(int(input_dim), dtype=np.float32)
    counts = np.zeros(int(input_dim), dtype=np.int64)
    sums = np.zeros(int(input_dim), dtype=np.float64)

    for sequence, valid_mask in zip(sequences, continuous_valid_masks):
        counts += np.asarray(valid_mask.sum(axis=0), dtype=np.int64)
        sums += np.asarray((sequence * valid_mask.astype(np.float32)).sum(axis=0), dtype=np.float64)

    valid_channels = counts > 0
    if np.any(valid_channels):
        means[valid_channels] = (sums[valid_channels] / counts[valid_channels]).astype(np.float32)

    sq_sums = np.zeros(int(input_dim), dtype=np.float64)
    for sequence, valid_mask in zip(sequences, continuous_valid_masks):
        centered = np.where(valid_mask, sequence - means[None, :], 0.0)
        sq_sums += np.square(centered, dtype=np.float64).sum(axis=0, dtype=np.float64)

    if np.any(valid_channels):
        std[valid_channels] = np.sqrt(
            np.maximum(sq_sums[valid_channels] / counts[valid_channels], 1e-6)
        ).astype(np.float32)
    std = np.where(std > 1e-6, std, np.ones_like(std, dtype=np.float32))

    return CNNNormalizationStats(
        channel_mean=means,
        channel_std=std,
    )


def _apply_normalization(
    *,
    sequences: tuple[np.ndarray, ...],
    continuous_valid_masks: tuple[np.ndarray, ...],
    normalization_stats: CNNNormalizationStats,
    channel_layout: CNNChannelLayout,
) -> tuple[np.ndarray, ...]:
    continuous_indices = np.asarray(channel_layout.continuous_indices, dtype=np.int64)
    normalized_sequences: list[np.ndarray] = []

    for sequence, valid_mask in zip(sequences, continuous_valid_masks):
        resolved = np.asarray(sequence, dtype=np.float32).copy()
        if continuous_indices.size > 0:
            resolved[:, continuous_indices] = (
                resolved[:, continuous_indices]
                - normalization_stats.channel_mean[continuous_indices][None, :]
            ) / normalization_stats.channel_std[continuous_indices][None, :]
            resolved[:, continuous_indices] = np.where(
                valid_mask[:, continuous_indices],
                resolved[:, continuous_indices],
                0.0,
            )
        normalized_sequences.append(resolved.astype(np.float32, copy=False))

    return tuple(normalized_sequences)


def build_per_layer_vectors(
    bundle: SupervisedFeatureBundle,
    normalization_stats: CNNNormalizationStats | None = None,
    *,
    include_total_layer_count: bool = True,
    depth_feature_mode: str = "both",
) -> CNNLayerVectorBatch:
    (
        values,
        group_mask,
        value_mask,
        depth_labels,
        slot_names,
        feature_names,
    ) = _resolve_bundle_metadata(bundle)
    retained_slot_indices = _resolve_retained_slot_indices(
        slot_names,
        group_mask=group_mask,
        value_mask=value_mask,
    )
    retained_slot_names = [slot_names[idx] for idx in retained_slot_indices]
    channel_layout = _build_channel_layout(
        retained_slot_names=retained_slot_names,
        feature_names=feature_names,
        include_total_layer_count=bool(include_total_layer_count),
        depth_feature_mode=str(depth_feature_mode),
    )

    retained_kind_array = np.asarray(
        [str(slot_name).partition(".")[0] for slot_name in retained_slot_names],
        dtype=object,
    )
    self_slot_mask = retained_kind_array == "self"
    cross_slot_mask = retained_kind_array == "cross"

    sequences: list[np.ndarray] = []
    continuous_valid_masks: list[np.ndarray] = []
    layer_names: list[tuple[str, ...]] = []

    for sample_idx in range(int(values.shape[0])):
        valid_depth_indices = np.flatnonzero(group_mask[int(sample_idx)])
        if valid_depth_indices.size == 0:
            raise ValueError(
                f"cnn_1d sample at row index {sample_idx} has zero valid layers after applying group_mask"
            )
        sample_depth_labels = [depth_labels[int(depth_idx)] for depth_idx in valid_depth_indices.tolist()]
        parsed_depths = [_parse_depth_label(depth_label) for depth_label in sample_depth_labels]
        arch_layer_max: dict[str, int] = {}
        for architecture_block, layer_index in parsed_depths:
            current_max = arch_layer_max.get(str(architecture_block))
            if current_max is None or int(layer_index) > current_max:
                arch_layer_max[str(architecture_block)] = int(layer_index)

        total_valid_layers = int(valid_depth_indices.size)
        sample_sequence = np.zeros((total_valid_layers, channel_layout.input_dim), dtype=np.float32)
        sample_continuous_valid = np.zeros_like(sample_sequence, dtype=bool)

        for sequence_index, depth_idx in enumerate(valid_depth_indices.tolist()):
            retained_values = np.asarray(
                values[int(sample_idx), int(depth_idx), retained_slot_indices, :],
                dtype=np.float32,
            )
            retained_value_mask = np.asarray(
                value_mask[int(sample_idx), int(depth_idx), retained_slot_indices, :],
                dtype=bool,
            )
            flat_values = retained_values.reshape(-1)
            flat_value_mask = retained_value_mask.reshape(-1)

            row = sample_sequence[int(sequence_index)]
            row[channel_layout.value_start : channel_layout.value_end] = np.where(
                flat_value_mask,
                flat_values,
                0.0,
            )
            sample_continuous_valid[
                int(sequence_index),
                channel_layout.value_start : channel_layout.value_end,
            ] = flat_value_mask

            slot_present = retained_value_mask.any(axis=1).astype(np.float32, copy=False)
            row[channel_layout.value_mask_start : channel_layout.value_mask_end] = flat_value_mask.astype(
                np.float32,
                copy=False,
            )
            row[channel_layout.slot_presence_start : channel_layout.slot_presence_end] = slot_present
            row[channel_layout.self_attention_index] = float(
                bool(np.any(slot_present[self_slot_mask])) if self_slot_mask.size > 0 else False
            )
            row[channel_layout.cross_attention_index] = float(
                bool(np.any(slot_present[cross_slot_mask])) if cross_slot_mask.size > 0 else False
            )

            architecture_block, layer_index = parsed_depths[int(sequence_index)]
            row[channel_layout.is_encoder_index] = float(architecture_block == "encoder")
            row[channel_layout.is_decoder_index] = float(architecture_block == "decoder")

            if channel_layout.layer_index_index is not None:
                row[int(channel_layout.layer_index_index)] = float(layer_index)
                sample_continuous_valid[int(sequence_index), int(channel_layout.layer_index_index)] = True
            if channel_layout.normalized_arch_index_index is not None:
                arch_max_index = int(arch_layer_max.get(str(architecture_block), 0))
                row[int(channel_layout.normalized_arch_index_index)] = _normalized_position(
                    int(layer_index),
                    int(arch_max_index),
                )
                sample_continuous_valid[
                    int(sequence_index),
                    int(channel_layout.normalized_arch_index_index),
                ] = True
            if channel_layout.normalized_sequence_index_index is not None:
                row[int(channel_layout.normalized_sequence_index_index)] = _normalized_position(
                    int(sequence_index),
                    int(total_valid_layers - 1),
                )
                sample_continuous_valid[
                    int(sequence_index),
                    int(channel_layout.normalized_sequence_index_index),
                ] = True
            if channel_layout.total_layers_index is not None:
                row[int(channel_layout.total_layers_index)] = float(total_valid_layers)
                sample_continuous_valid[
                    int(sequence_index),
                    int(channel_layout.total_layers_index),
                ] = True

        sequences.append(sample_sequence)
        continuous_valid_masks.append(sample_continuous_valid)
        layer_names.append(tuple(sample_depth_labels))

    resolved_stats = normalization_stats
    if resolved_stats is None:
        resolved_stats = _compute_normalization_stats(
            sequences=tuple(sequences),
            continuous_valid_masks=tuple(continuous_valid_masks),
            input_dim=int(channel_layout.input_dim),
        )
    normalized_sequences = _apply_normalization(
        sequences=tuple(sequences),
        continuous_valid_masks=tuple(continuous_valid_masks),
        normalization_stats=resolved_stats,
        channel_layout=channel_layout,
    )

    return CNNLayerVectorBatch(
        sequences=normalized_sequences,
        continuous_valid_masks=tuple(continuous_valid_masks),
        layer_names=tuple(layer_names),
        channel_layout=channel_layout,
        normalization_stats=resolved_stats,
    )


def pad_layer_sequence_batch(sequences: tuple[np.ndarray, ...] | list[np.ndarray]) -> CNNFeatureTensors:
    if not sequences:
        raise ValueError("cnn_1d requires at least one layer sequence to build a batch")

    resolved_sequences = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
    if resolved_sequences[0].ndim != 2:
        raise ValueError(
            f"cnn_1d expected 2D layer sequences, got shape={resolved_sequences[0].shape}"
        )
    input_dim = int(resolved_sequences[0].shape[1])
    max_layers = 0
    for sequence in resolved_sequences:
        if sequence.ndim != 2:
            raise ValueError(f"cnn_1d expected 2D layer sequences, got shape={sequence.shape}")
        if int(sequence.shape[1]) != input_dim:
            raise ValueError("cnn_1d layer sequences must all share the same feature dimension")
        if int(sequence.shape[0]) <= 0:
            raise ValueError("cnn_1d does not support empty layer sequences")
        max_layers = max(max_layers, int(sequence.shape[0]))

    batch_inputs = np.zeros((len(resolved_sequences), max_layers, input_dim), dtype=np.float32)
    layer_mask = np.zeros((len(resolved_sequences), max_layers), dtype=np.float32)
    for sample_idx, sequence in enumerate(resolved_sequences):
        n_layers = int(sequence.shape[0])
        batch_inputs[int(sample_idx), :n_layers, :] = sequence
        layer_mask[int(sample_idx), :n_layers] = 1.0

    return CNNFeatureTensors(inputs=batch_inputs, layer_mask=layer_mask)


def masked_mean_pool(hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
    mask = layer_mask.unsqueeze(1).to(dtype=hidden.dtype)
    valid_counts = torch.clamp(mask.sum(dim=2), min=1.0)
    return (hidden * mask).sum(dim=2) / valid_counts


def masked_max_pool(hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
    mask = layer_mask.unsqueeze(1) > 0
    masked_hidden = hidden.masked_fill(~mask, float("-inf"))
    pooled = masked_hidden.max(dim=2).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class _LayerNorm1d(TorchModuleBase):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.norm = nn.LayerNorm(int(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _Conv1DBlock(TorchModuleBase):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        normalization: str,
        use_residual: bool,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        padding = max(0, (int(kernel_size) - 1) * int(dilation) // 2)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.use_residual = bool(use_residual)
        self.conv = nn.Conv1d(
            int(input_channels),
            int(output_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            dilation=int(dilation),
        )
        if str(normalization) == "layernorm":
            self.normalization = _LayerNorm1d(int(output_channels))
        elif str(normalization) == "batchnorm":
            self.normalization = nn.BatchNorm1d(int(output_channels))
        elif str(normalization) == "none":
            self.normalization = None
        else:
            raise ValueError(
                "cnn_1d normalization must be one of {'layernorm', 'batchnorm', 'none'}"
            )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def _propagate_mask(self, layer_mask: torch.Tensor) -> torch.Tensor:
        assert F is not None
        pooled_mask = F.max_pool1d(
            layer_mask.unsqueeze(1),
            kernel_size=int(self.kernel_size),
            stride=int(self.stride),
            padding=int(self.padding),
            dilation=int(self.dilation),
        )
        return (pooled_mask.squeeze(1) > 0).to(dtype=layer_mask.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.conv(hidden)
        next_layer_mask = self._propagate_mask(layer_mask)
        if self.normalization is not None:
            next_hidden = self.normalization(next_hidden)
        next_hidden = self.activation(next_hidden)
        next_hidden = self.dropout(next_hidden)

        mask = next_layer_mask.unsqueeze(1).to(dtype=next_hidden.dtype)
        next_hidden = next_hidden * mask
        if (
            self.use_residual
            and hidden.shape == next_hidden.shape
            and tuple(layer_mask.shape) == tuple(next_layer_mask.shape)
        ):
            next_hidden = (next_hidden + hidden * mask) * mask
        return next_hidden, next_layer_mask


class Conv1DLayerAggregator(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        if str(config.pooling) not in {"mean", "max", "mean_max"}:
            raise ValueError("cnn_1d pooling must be one of {'mean', 'max', 'mean_max'}")

        blocks: list[_Conv1DBlock] = []
        in_channels = int(input_dim)
        for _ in range(int(config.num_conv_layers)):
            blocks.append(
                _Conv1DBlock(
                    input_channels=int(in_channels),
                    output_channels=int(config.conv_channels),
                    kernel_size=int(config.kernel_size),
                    stride=int(config.stride),
                    dilation=int(config.dilation),
                    dropout=float(config.dropout),
                    normalization=str(config.normalization),
                    use_residual=bool(config.use_residual),
                )
            )
            in_channels = int(config.conv_channels)
        self.blocks = nn.ModuleList(blocks)
        self.pooling = str(config.pooling)
        self.output_channels = int(config.conv_channels)
        self.embedding_dim = (
            int(config.conv_channels) * 2
            if str(config.pooling) == "mean_max"
            else int(config.conv_channels)
        )

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        hidden = inputs.transpose(1, 2)
        current_mask = layer_mask
        for block in self.blocks:
            hidden, current_mask = block(hidden, current_mask)
        hidden = hidden * current_mask.unsqueeze(1).to(dtype=hidden.dtype)

        if self.pooling == "mean":
            return masked_mean_pool(hidden, current_mask)
        if self.pooling == "max":
            return masked_max_pool(hidden, current_mask)
        return torch.cat(
            [
                masked_mean_pool(hidden, current_mask),
                masked_max_pool(hidden, current_mask),
            ],
            dim=1,
        )


class _CNNLayerSequenceClassifier(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
        output_dim: int,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.aggregator = Conv1DLayerAggregator(
            input_dim=int(input_dim),
            config=config,
        )
        self.output_dim = int(output_dim)
        self.head = nn.Linear(int(self.aggregator.embedding_dim), int(self.output_dim))

    def extract_features(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        return self.aggregator(inputs, layer_mask)

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_features(inputs, layer_mask)
        logits = self.head(embedding)
        if self.output_dim == 1:
            return logits.squeeze(1)
        return logits


class _GradientReverseFunction(torch.autograd.Function if torch is not None else object):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -float(ctx.lambda_value) * grad_output, None


def _gradient_reverse(inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
    _require_torch()
    return _GradientReverseFunction.apply(inputs, float(lambda_value))


class _CNNLayerSequenceDANNClassifier(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
        output_dim: int,
        domain_output_dim: int,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.aggregator = Conv1DLayerAggregator(
            input_dim=int(input_dim),
            config=config,
        )
        self.output_dim = int(output_dim)
        self.domain_output_dim = int(domain_output_dim)
        if self.domain_output_dim < 2:
            raise ValueError("cnn_1d_dann requires at least two rank-domain classes")
        embedding_dim = int(self.aggregator.embedding_dim)
        self.head = nn.Linear(embedding_dim, int(self.output_dim))
        self.domain_head = nn.Linear(embedding_dim, int(self.domain_output_dim))

    def extract_features(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        return self.aggregator(inputs, layer_mask)

    def forward(
        self,
        inputs: torch.Tensor,
        layer_mask: torch.Tensor,
        *,
        lambda_value: float | None = None,
        return_domain: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.extract_features(inputs, layer_mask)
        label_logits = self.head(embedding)
        if self.output_dim == 1:
            label_logits = label_logits.squeeze(1)
        if not return_domain:
            return label_logits
        reversed_embedding = _gradient_reverse(embedding, float(lambda_value or 0.0))
        domain_logits = self.domain_head(reversed_embedding)
        return label_logits, domain_logits


class CNN1DSupervisedModel:
    def __init__(
        self,
        *,
        conv_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float,
        use_residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean_max",
        include_total_layer_count: bool = True,
        depth_feature_mode: str = "both",
        learning_rate: float,
        weight_decay: float,
        random_state: int,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = CNN_MAX_EPOCHS,
        batch_size: int = CNN_BATCH_SIZE,
        patience: int = CNN_PATIENCE,
        class_weight_loss: bool = False,
        rank_label_weight_loss: bool = False,
    ) -> None:
        _require_torch()
        if int(conv_channels) <= 0:
            raise ValueError("cnn_1d conv_channels must be positive")
        if int(num_conv_layers) <= 0:
            raise ValueError("cnn_1d num_conv_layers must be positive")
        if int(kernel_size) <= 0:
            raise ValueError("cnn_1d kernel_size must be positive")
        if int(stride) <= 0:
            raise ValueError("cnn_1d stride must be positive")
        if int(dilation) <= 0:
            raise ValueError("cnn_1d dilation must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("cnn_1d dropout must be in [0, 1)")
        if bool(class_weight_loss) and bool(rank_label_weight_loss):
            raise ValueError("cnn_1d supports either class_weight_loss or rank_label_weight_loss, not both")
        self.layer_vector_config = CNNLayerVectorConfig(
            conv_channels=int(conv_channels),
            num_conv_layers=int(num_conv_layers),
            kernel_size=int(kernel_size),
            stride=int(stride),
            dilation=int(dilation),
            dropout=float(dropout),
            use_residual=bool(use_residual),
            normalization=str(normalization),
            pooling=str(pooling),
            include_total_layer_count=bool(include_total_layer_count),
            depth_feature_mode=str(depth_feature_mode),
        )
        self.conv_channels = int(conv_channels)
        self.num_conv_layers = int(num_conv_layers)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)
        self.normalization = str(normalization)
        self.pooling = str(pooling)
        self.include_total_layer_count = bool(include_total_layer_count)
        self.depth_feature_mode = str(depth_feature_mode)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.task_spec = task_spec if task_spec is not None else _default_binary_task_spec()
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.class_weight_loss = bool(class_weight_loss)
        self.rank_label_weight_loss = bool(rank_label_weight_loss)
        self.model_: _CNNLayerSequenceClassifier | None = None
        self.normalization_stats_: CNNNormalizationStats | None = None
        self.channel_layout_: CNNChannelLayout | None = None
        self.channel_mean_: np.ndarray | None = None
        self.channel_std_: np.ndarray | None = None
        self.input_channels_: int | None = None
        self.classes_ = np.arange(self.task_spec.n_classes, dtype=np.int32)
        self.class_names_ = tuple(str(x) for x in self.task_spec.class_names)
        self.backend_name_ = "cnn_1d"
        self._fit_summary: dict[str, Any] = {}

    def _set_random_seeds(self) -> None:
        assert torch is not None
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def _set_threads(self, n_jobs: int | None) -> None:
        assert torch is not None
        if n_jobs is None:
            return
        resolved = int(n_jobs)
        if resolved <= 0:
            return
        torch.set_num_threads(resolved)

    def _prepare_numpy_inputs(self, bundle: SupervisedFeatureBundle) -> CNNFeatureTensors:
        vector_batch = build_per_layer_vectors(
            bundle,
            normalization_stats=self.normalization_stats_,
            include_total_layer_count=bool(self.include_total_layer_count),
            depth_feature_mode=str(self.depth_feature_mode),
        )
        if self.channel_layout_ is None:
            self.channel_layout_ = vector_batch.channel_layout
        elif self.channel_layout_ != vector_batch.channel_layout:
            raise ValueError(
                "cnn_1d encountered an incompatible layer-sequence channel layout between bundles"
            )

        if self.normalization_stats_ is None:
            self.normalization_stats_ = vector_batch.normalization_stats
            self.channel_mean_ = np.asarray(
                vector_batch.normalization_stats.channel_mean,
                dtype=np.float32,
            )
            self.channel_std_ = np.asarray(
                vector_batch.normalization_stats.channel_std,
                dtype=np.float32,
            )
        elif (
            self.channel_mean_ is None
            or self.channel_std_ is None
        ):
            self.channel_mean_ = np.asarray(
                self.normalization_stats_.channel_mean,
                dtype=np.float32,
            )
            self.channel_std_ = np.asarray(
                self.normalization_stats_.channel_std,
                dtype=np.float32,
            )

        padded_batch = pad_layer_sequence_batch(vector_batch.sequences)
        self.input_channels_ = int(padded_batch.inputs.shape[2])
        return padded_batch

    def _build_model(self, *, input_channels: int) -> _CNNLayerSequenceClassifier:
        _require_torch()
        model = _CNNLayerSequenceClassifier(
            input_dim=int(input_channels),
            config=self.layer_vector_config,
            output_dim=(1 if self.task_spec.is_binary else self.task_spec.n_classes),
        )
        self.input_channels_ = int(input_channels)
        return model

    def _logits_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                logits = self.model_(batch_inputs, batch_layer_mask)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        if not outputs:
            if self.task_spec.is_binary:
                return np.asarray([], dtype=np.float64)
            return np.asarray([], dtype=np.float64).reshape(0, self.task_spec.n_classes)
        combined = np.concatenate(outputs, axis=0)
        if self.task_spec.is_binary:
            return combined.reshape(-1)
        return np.asarray(combined, dtype=np.float64)

    def _features_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, batch_layer_mask in loader:
                embeddings = self.model_.extract_features(batch_inputs, batch_layer_mask)
                outputs.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
        if not outputs:
            embedding_dim = 0
            if self.model_ is not None:
                embedding_dim = int(self.model_.aggregator.embedding_dim)
            return np.asarray([], dtype=np.float32).reshape(0, embedding_dim)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        rank_labels: np.ndarray | None = None,
    ) -> "CNN1DSupervisedModel":
        _require_torch()
        assert torch is not None
        self._set_random_seeds()
        self._set_threads(n_jobs)

        train_tensors = self._prepare_numpy_inputs(bundle)
        if self.task_spec.is_binary:
            y_train = np.asarray(labels, dtype=np.float32).reshape(-1)
        else:
            y_train = np.asarray(labels, dtype=np.int64).reshape(-1)
        if train_tensors.inputs.shape[0] != y_train.shape[0]:
            raise ValueError("cnn_1d training features/labels length mismatch")
        rank_label_loss_config: dict[str, Any] | None = None
        rank_label_sample_weights = np.ones(y_train.shape[0], dtype=np.float32)
        if self.rank_label_weight_loss:
            if rank_labels is None:
                raise ValueError("cnn_1d rank_label_weight_loss requires rank_labels")
            rank_labels_np = np.asarray(rank_labels, dtype=np.int64).reshape(-1)
            if rank_labels_np.shape[0] != y_train.shape[0]:
                raise ValueError("cnn_1d rank_labels must have the same length as labels")
            rank_label_sample_weights, rank_label_loss_config = compute_balanced_rank_label_loss_config(
                y_train,
                rank_labels_np,
                task_spec=self.task_spec,
            )

        train_dataset = TensorDataset(
            torch.from_numpy(train_tensors.inputs),
            torch.from_numpy(train_tensors.layer_mask),
            torch.from_numpy(y_train),
            torch.from_numpy(rank_label_sample_weights),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=True,
        )

        valid_loader = None
        y_valid = None
        if validation_data is not None:
            valid_bundle, valid_labels = validation_data
            valid_tensors = self._prepare_numpy_inputs(valid_bundle)
            y_valid = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
            valid_dataset = TensorDataset(
                torch.from_numpy(valid_tensors.inputs),
                torch.from_numpy(valid_tensors.layer_mask),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=min(self.batch_size, max(1, len(valid_dataset))),
                shuffle=False,
            )

        self.model_ = self._build_model(input_channels=int(train_tensors.inputs.shape[2]))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        class_loss_config: dict[str, Any] | None = None
        if self.class_weight_loss:
            class_loss_config = compute_balanced_class_loss_config(
                y_train,
                task_spec=self.task_spec,
            )

        if self.task_spec.is_binary:
            pos_weight = None
            if class_loss_config is not None:
                pos_weight = torch.tensor(
                    float(class_loss_config["binary_pos_weight"]),
                    dtype=torch.float32,
                )
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        else:
            class_weight_tensor = None
            if class_loss_config is not None:
                class_weight_tensor = torch.tensor(
                    np.asarray(class_loss_config["class_weights"], dtype=np.float32),
                    dtype=torch.float32,
                )
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weight_tensor,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )

        best_state = None
        best_metric = -math.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch_idx in range(self.max_epochs):
            self.model_.train()
            train_loss_sum = 0.0
            train_count = 0.0
            for batch_inputs, batch_layer_mask, batch_labels, batch_sample_weights in train_loader:
                optimizer.zero_grad()
                logits = self.model_(batch_inputs, batch_layer_mask)
                loss_raw = loss_fn(logits, batch_labels)
                if self.rank_label_weight_loss:
                    weights = batch_sample_weights.to(dtype=loss_raw.dtype)
                    loss_numerator = torch.sum(loss_raw.reshape(-1) * weights.reshape(-1))
                    loss_denominator = torch.clamp(torch.sum(weights), min=1.0)
                    loss = loss_numerator / loss_denominator
                else:
                    loss = loss_raw
                loss.backward()
                optimizer.step()
                batch_size = int(batch_labels.shape[0])
                if self.rank_label_weight_loss:
                    train_loss_sum += float(loss_numerator.item())
                    train_count += float(loss_denominator.item())
                else:
                    train_loss_sum += float(loss.item()) * batch_size
                    train_count += float(batch_size)

            train_loss = train_loss_sum / max(1, train_count)
            metric = -train_loss
            if valid_loader is not None and y_valid is not None:
                valid_logits = self._logits_from_loader(valid_loader)
                if self.task_spec.is_binary:
                    metric = float(roc_auc_score(y_valid, valid_logits))
                else:
                    valid_pred = np.argmax(valid_logits, axis=1).astype(np.int32, copy=False)
                    metric = float(f1_score(y_valid, valid_pred, average="macro"))

            history.append(
                {
                    "epoch": float(epoch_idx),
                    "train_loss": float(train_loss),
                    "selection_metric": float(metric),
                    "selection_metric_name": str(self.task_spec.selection_metric_name),
                }
            )

            if metric > best_metric:
                best_metric = float(metric)
                best_epoch = int(epoch_idx)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if valid_loader is not None and stale_epochs >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("cnn_1d training did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": float(best_metric),
            "selection_metric_name": str(self.task_spec.selection_metric_name),
            "epochs_ran": int(len(history)),
            "history": history,
            "class_weight_loss": bool(self.class_weight_loss),
            "class_loss_weights": class_loss_config,
            "rank_label_weight_loss": bool(self.rank_label_weight_loss),
            "rank_label_loss_weights": rank_label_loss_config,
        }
        return self

    def decision_function(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        return self._logits_from_loader(loader)

    @property
    def fit_summary(self) -> dict[str, Any]:
        """Get the fit summary containing training details and history."""
        if not hasattr(self, "_fit_summary") or not self._fit_summary:
            return {}
        return dict(self._fit_summary)

    def extract_features(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("cnn_1d model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        dataset = TensorDataset(
            torch.from_numpy(tensors.inputs),
            torch.from_numpy(tensors.layer_mask),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=False,
        )
        return self._features_from_loader(loader)

    def predict_proba(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        logits = self.decision_function(bundle)
        if self.task_spec.is_binary:
            probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64).reshape(-1)))
            return np.column_stack([1.0 - probabilities, probabilities]).astype(np.float64, copy=False)
        logits_2d = np.asarray(logits, dtype=np.float64)
        shifted = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        normalizer = np.sum(exp_logits, axis=1, keepdims=True)
        normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
        return np.asarray(exp_logits / normalizer, dtype=np.float64)

    def predict(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        if self.task_spec.is_binary:
            return (self.decision_function(bundle) >= 0.0).astype(np.int32, copy=False)
        probabilities = self.predict_proba(bundle)
        return np.argmax(probabilities, axis=1).astype(np.int32, copy=False)

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {}

    def save(self, path: Path) -> None:
        _require_torch()
        assert torch is not None
        if (
            self.model_ is None
            or self.channel_mean_ is None
            or self.channel_std_ is None
            or self.channel_layout_ is None
        ):
            raise RuntimeError("cnn_1d model has not been fit")
        payload = {
            "backend": self.backend_name_,
            "config": {
                "conv_channels": int(self.conv_channels),
                "num_conv_layers": int(self.num_conv_layers),
                "kernel_size": int(self.kernel_size),
                "stride": int(self.stride),
                "dilation": int(self.dilation),
                "dropout": float(self.dropout),
                "use_residual": bool(self.use_residual),
                "normalization": str(self.normalization),
                "pooling": str(self.pooling),
                "include_total_layer_count": bool(self.include_total_layer_count),
                "depth_feature_mode": str(self.depth_feature_mode),
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
                "random_state": int(self.random_state),
                "max_epochs": int(self.max_epochs),
                "batch_size": int(self.batch_size),
                "patience": int(self.patience),
                "class_weight_loss": bool(self.class_weight_loss),
                "rank_label_weight_loss": bool(self.rank_label_weight_loss),
                "input_channels": int(self.input_channels_ or 0),
                "task_mode": str(self.task_spec.task_mode),
                "num_classes": int(self.task_spec.n_classes),
            },
            "channel_layout": asdict(self.channel_layout_),
            "state_dict": self.model_.state_dict(),
            "normalization": {
                "channel_mean": np.asarray(self.channel_mean_, dtype=np.float32),
                "channel_std": np.asarray(self.channel_std_, dtype=np.float32),
            },
            "classes": np.asarray(self.classes_, dtype=np.int32),
            "class_names": list(self.class_names_),
            "task": self.task_spec.to_dict(),
            "fit_summary": dict(self._fit_summary),
        }
        payload.update(self._checkpoint_extra_payload())
        torch.save(payload, Path(path).expanduser().resolve())


class CNN1DDANNSupervisedModel(CNN1DSupervisedModel):
    def __init__(
        self,
        *,
        conv_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float,
        use_residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean_max",
        include_total_layer_count: bool = True,
        depth_feature_mode: str = "both",
        learning_rate: float,
        weight_decay: float,
        random_state: int,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = CNN_MAX_EPOCHS,
        batch_size: int = CNN_BATCH_SIZE,
        patience: int = CNN_PATIENCE,
        class_weight_loss: bool = False,
        rank_label_weight_loss: bool = False,
        source_rank: int = 256,
        dann_lambda_max: float = 1.0,
        dann_lambda_gamma: float = 10.0,
        dann_lr_alpha: float = 10.0,
        dann_lr_beta: float = 0.75,
    ) -> None:
        super().__init__(
            conv_channels=conv_channels,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            use_residual=use_residual,
            normalization=normalization,
            pooling=pooling,
            include_total_layer_count=include_total_layer_count,
            depth_feature_mode=depth_feature_mode,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            task_spec=task_spec,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            class_weight_loss=class_weight_loss,
            rank_label_weight_loss=rank_label_weight_loss,
        )
        if float(dann_lambda_max) < 0.0:
            raise ValueError("cnn_1d_dann dann_lambda_max must be non-negative")
        if float(dann_lambda_gamma) < 0.0:
            raise ValueError("cnn_1d_dann dann_lambda_gamma must be non-negative")
        if float(dann_lr_alpha) < 0.0:
            raise ValueError("cnn_1d_dann dann_lr_alpha must be non-negative")
        if float(dann_lr_beta) < 0.0:
            raise ValueError("cnn_1d_dann dann_lr_beta must be non-negative")
        self.source_rank = int(source_rank)
        self.dann_lambda_max = float(dann_lambda_max)
        self.dann_lambda_gamma = float(dann_lambda_gamma)
        self.dann_lr_alpha = float(dann_lr_alpha)
        self.dann_lr_beta = float(dann_lr_beta)
        self.backend_name_ = "cnn_1d_dann"
        self.domain_classes_: np.ndarray | None = None
        self.domain_class_names_: tuple[str, ...] = ()
        self.domain_rank_values_: tuple[int, ...] = ()
        self.domain_class_to_index_: dict[str, int] = {}

    def _build_model(self, *, input_channels: int) -> _CNNLayerSequenceDANNClassifier:
        _require_torch()
        if self.domain_classes_ is None or int(self.domain_classes_.shape[0]) < 2:
            raise ValueError("cnn_1d_dann requires domain classes before building the model")
        model = _CNNLayerSequenceDANNClassifier(
            input_dim=int(input_channels),
            config=self.layer_vector_config,
            output_dim=(1 if self.task_spec.is_binary else self.task_spec.n_classes),
            domain_output_dim=int(self.domain_classes_.shape[0]),
        )
        self.input_channels_ = int(input_channels)
        return model

    def _dann_lambda(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return float(
            self.dann_lambda_max
            * (2.0 / (1.0 + math.exp(-float(self.dann_lambda_gamma) * p)) - 1.0)
        )

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        domain_labels: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        rank_labels: np.ndarray | None = None,
        domain_class_names: list[str] | tuple[str, ...] | None = None,
        domain_rank_values: list[int] | tuple[int, ...] | np.ndarray | None = None,
    ) -> "CNN1DDANNSupervisedModel":
        _require_torch()
        assert torch is not None
        self._set_random_seeds()
        self._set_threads(n_jobs)

        if domain_labels is None:
            raise ValueError("cnn_1d_dann requires domain_labels for rank-adversarial training")

        train_tensors = self._prepare_numpy_inputs(bundle)
        y_raw = np.asarray(labels).reshape(-1)
        domain_np = np.asarray(domain_labels, dtype=np.int64).reshape(-1)
        n_rows = int(train_tensors.inputs.shape[0])
        if label_mask is None:
            label_mask_np = np.ones(n_rows, dtype=bool)
            label_loss_scope = "all_training_ranks"
        else:
            label_mask_np = np.asarray(label_mask, dtype=bool).reshape(-1)
            label_loss_scope = (
                "all_training_ranks"
                if bool(np.all(label_mask_np))
                else "masked_training_rows"
            )
        if y_raw.shape[0] != n_rows or domain_np.shape[0] != n_rows or label_mask_np.shape[0] != n_rows:
            raise ValueError("cnn_1d_dann training features, labels, domains, and masks length mismatch")
        if not bool(np.any(label_mask_np)):
            raise ValueError("cnn_1d_dann requires at least one labeled training row")
        rank_label_loss_config: dict[str, Any] | None = None
        rank_label_sample_weights = np.ones(n_rows, dtype=np.float32)
        if self.rank_label_weight_loss:
            if rank_labels is None:
                raise ValueError("cnn_1d_dann rank_label_weight_loss requires rank_labels")
            rank_labels_np = np.asarray(rank_labels, dtype=np.int64).reshape(-1)
            if rank_labels_np.shape[0] != n_rows:
                raise ValueError("cnn_1d_dann rank_labels must have the same length as labels")
            rank_label_sample_weights, rank_label_loss_config = compute_balanced_rank_label_loss_config(
                y_raw,
                rank_labels_np,
                task_spec=self.task_spec,
                sample_mask=label_mask_np,
            )

        observed_domain_classes = np.unique(domain_np)
        if observed_domain_classes.size < 2:
            raise ValueError("cnn_1d_dann requires at least two observed rank-domain classes")
        expected_classes = np.arange(int(np.max(observed_domain_classes)) + 1, dtype=np.int64)
        if not np.array_equal(observed_domain_classes, expected_classes):
            raise ValueError(
                "cnn_1d_dann domain_labels must be contiguous integer class ids starting at zero"
            )
        self.domain_classes_ = expected_classes.astype(np.int64, copy=False)
        if domain_class_names is None:
            self.domain_class_names_ = tuple(f"domain_{idx}" for idx in self.domain_classes_.tolist())
        else:
            names = tuple(str(x) for x in domain_class_names)
            if len(names) != int(self.domain_classes_.shape[0]):
                raise ValueError("cnn_1d_dann domain_class_names length must match domain classes")
            self.domain_class_names_ = names
        if domain_rank_values is None:
            self.domain_rank_values_ = tuple(int(x) for x in self.domain_classes_.tolist())
        else:
            ranks = tuple(int(x) for x in np.asarray(domain_rank_values, dtype=np.int64).reshape(-1).tolist())
            if len(ranks) != int(self.domain_classes_.shape[0]):
                raise ValueError("cnn_1d_dann domain_rank_values length must match domain classes")
            self.domain_rank_values_ = ranks
        self.domain_class_to_index_ = {
            name: int(idx) for idx, name in enumerate(self.domain_class_names_)
        }

        class_loss_config: dict[str, Any] | None = None
        if self.class_weight_loss:
            class_loss_config = compute_balanced_class_loss_config(
                y_raw,
                task_spec=self.task_spec,
                sample_mask=label_mask_np,
            )

        if self.task_spec.is_binary:
            y_train = np.asarray(y_raw, dtype=np.float32).reshape(-1)
            pos_weight = None
            if class_loss_config is not None:
                pos_weight = torch.tensor(
                    float(class_loss_config["binary_pos_weight"]),
                    dtype=torch.float32,
                )
            label_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        else:
            y_train = np.asarray(y_raw, dtype=np.int64).reshape(-1)
            class_weight_tensor = None
            if class_loss_config is not None:
                class_weight_tensor = torch.tensor(
                    np.asarray(class_loss_config["class_weights"], dtype=np.float32),
                    dtype=torch.float32,
                )
            label_loss_fn = nn.CrossEntropyLoss(
                weight=class_weight_tensor,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        domain_loss_fn = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(
            torch.from_numpy(train_tensors.inputs),
            torch.from_numpy(train_tensors.layer_mask),
            torch.from_numpy(y_train),
            torch.from_numpy(domain_np),
            torch.from_numpy(label_mask_np.astype(np.float32)),
            torch.from_numpy(rank_label_sample_weights),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=True,
        )

        valid_loader = None
        y_valid = None
        if validation_data is not None:
            valid_bundle, valid_labels = validation_data
            valid_tensors = self._prepare_numpy_inputs(valid_bundle)
            y_valid = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
            valid_dataset = TensorDataset(
                torch.from_numpy(valid_tensors.inputs),
                torch.from_numpy(valid_tensors.layer_mask),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=min(self.batch_size, max(1, len(valid_dataset))),
                shuffle=False,
            )

        self.model_ = self._build_model(input_channels=int(train_tensors.inputs.shape[2]))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        best_state = None
        best_metric = -math.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, float]] = []
        total_steps = max(1, int(self.max_epochs) * max(1, len(train_loader)))
        global_step = 0

        for epoch_idx in range(self.max_epochs):
            self.model_.train()
            train_loss_sum = 0.0
            label_loss_sum = 0.0
            domain_loss_sum = 0.0
            train_count = 0
            label_count = 0.0
            last_lambda = 0.0
            last_lr = float(self.learning_rate)

            for (
                batch_inputs,
                batch_layer_mask,
                batch_labels,
                batch_domains,
                batch_label_mask,
                batch_label_weights,
            ) in train_loader:
                progress = float(global_step) / float(max(1, total_steps - 1))
                lambda_value = self._dann_lambda(progress)
                current_lr = float(self.learning_rate)

                optimizer.zero_grad()
                label_logits, domain_logits = self.model_(
                    batch_inputs,
                    batch_layer_mask,
                    lambda_value=float(lambda_value),
                    return_domain=True,
                )
                domain_loss = domain_loss_fn(domain_logits, batch_domains)

                supervised_mask = batch_label_mask > 0.5
                if bool(torch.any(supervised_mask)):
                    if self.task_spec.is_binary:
                        label_loss_raw = label_loss_fn(
                            label_logits[supervised_mask],
                            batch_labels[supervised_mask],
                        )
                    else:
                        label_loss_raw = label_loss_fn(
                            label_logits[supervised_mask],
                            batch_labels[supervised_mask].to(dtype=torch.long),
                        )
                    current_label_count = float(torch.sum(supervised_mask).item())
                    if self.rank_label_weight_loss:
                        selected_weights = batch_label_weights[supervised_mask].to(dtype=label_loss_raw.dtype)
                        label_loss_numerator = torch.sum(
                            label_loss_raw.reshape(-1) * selected_weights.reshape(-1)
                        )
                        label_loss_denominator = torch.clamp(torch.sum(selected_weights), min=1.0)
                        label_loss = label_loss_numerator / label_loss_denominator
                        current_label_count = float(label_loss_denominator.item())
                    else:
                        label_loss = label_loss_raw
                else:
                    label_loss = domain_loss.new_zeros(())
                    current_label_count = 0.0

                loss = label_loss + domain_loss
                loss.backward()
                optimizer.step()

                batch_size = int(batch_domains.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                domain_loss_sum += float(domain_loss.item()) * batch_size
                if current_label_count > 0:
                    if self.rank_label_weight_loss:
                        label_loss_sum += float(label_loss_numerator.item())
                    else:
                        label_loss_sum += float(label_loss.item()) * current_label_count
                    label_count += current_label_count
                train_count += batch_size
                last_lambda = float(lambda_value)
                last_lr = float(current_lr)
                global_step += 1

            train_loss = train_loss_sum / max(1, train_count)
            metric = -train_loss
            if valid_loader is not None and y_valid is not None:
                valid_logits = self._logits_from_loader(valid_loader)
                if self.task_spec.is_binary:
                    metric = float(roc_auc_score(y_valid, valid_logits))
                else:
                    valid_pred = np.argmax(valid_logits, axis=1).astype(np.int32, copy=False)
                    metric = float(f1_score(y_valid, valid_pred, average="macro"))

            history.append(
                {
                    "epoch": float(epoch_idx),
                    "train_loss": float(train_loss),
                    "label_loss": float(label_loss_sum / max(1, label_count)),
                    "domain_loss": float(domain_loss_sum / max(1, train_count)),
                    "label_rows": float(label_count),
                    "domain_rows": float(train_count),
                    "dann_lambda": float(last_lambda),
                    "learning_rate": float(last_lr),
                    "selection_metric": float(metric),
                    "selection_metric_name": str(self.task_spec.selection_metric_name),
                }
            )

            if metric > best_metric:
                best_metric = float(metric)
                best_epoch = int(epoch_idx)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if valid_loader is not None and stale_epochs >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("cnn_1d_dann training did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": float(best_metric),
            "selection_metric_name": str(self.task_spec.selection_metric_name),
            "epochs_ran": int(len(history)),
            "history": history,
            "domain_loss_weight": 1.0,
            "label_loss_scope": str(label_loss_scope),
            "class_weight_loss": bool(self.class_weight_loss),
            "class_loss_weights": class_loss_config,
            "rank_label_weight_loss": bool(self.rank_label_weight_loss),
            "rank_label_loss_weights": rank_label_loss_config,
        }
        return self

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {
            "domain_adaptation": {
                "source_rank": int(self.source_rank),
                "domain_class_names": list(self.domain_class_names_),
                "domain_rank_values": [int(x) for x in self.domain_rank_values_],
                "domain_class_to_index": dict(self.domain_class_to_index_),
                "label_loss_scope": str(self._fit_summary.get("label_loss_scope", "all_training_ranks")),
                "domain_loss": "multiclass_rank_cross_entropy",
                "domain_loss_weight": 1.0,
                "lambda_schedule": {
                    "type": "dann_paper_logistic",
                    "lambda_max": float(self.dann_lambda_max),
                    "gamma": float(self.dann_lambda_gamma),
                },
                "learning_rate_schedule": {
                    "type": "fixed",
                    "learning_rate": float(self.learning_rate),
                },
            }
        }


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    _require_torch()
    assert torch is not None
    resolved_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(resolved_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"CNN checkpoint must contain a dictionary payload: {resolved_path}")
    return payload


def _channel_layout_from_payload(payload: Any) -> CNNChannelLayout:
    if not isinstance(payload, dict):
        raise ValueError("CNN checkpoint is missing channel_layout")

    def required_int(name: str) -> int:
        if name not in payload:
            raise ValueError(f"CNN checkpoint channel_layout is missing {name!r}")
        return int(payload[name])

    def optional_int(name: str) -> int | None:
        value = payload.get(name)
        return None if value is None else int(value)

    return CNNChannelLayout(
        slot_names=tuple(str(x) for x in payload.get("slot_names", [])),
        feature_names=tuple(str(x) for x in payload.get("feature_names", [])),
        value_start=required_int("value_start"),
        value_end=required_int("value_end"),
        value_mask_start=required_int("value_mask_start"),
        value_mask_end=required_int("value_mask_end"),
        slot_presence_start=required_int("slot_presence_start"),
        slot_presence_end=required_int("slot_presence_end"),
        self_attention_index=required_int("self_attention_index"),
        cross_attention_index=required_int("cross_attention_index"),
        is_encoder_index=required_int("is_encoder_index"),
        is_decoder_index=required_int("is_decoder_index"),
        layer_index_index=optional_int("layer_index_index"),
        normalized_arch_index_index=optional_int("normalized_arch_index_index"),
        normalized_sequence_index_index=optional_int("normalized_sequence_index_index"),
        total_layers_index=optional_int("total_layers_index"),
        continuous_indices=tuple(int(x) for x in payload.get("continuous_indices", [])),
    )


def load_cnn_checkpoint(path: Path) -> CNN1DSupervisedModel:
    payload = _torch_load_checkpoint(path)
    backend = str(payload.get("backend") or "cnn_1d")
    if backend not in {"cnn_1d", "cnn_1d_dann"}:
        raise ValueError(f"Unsupported CNN checkpoint backend={backend!r}")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CNN checkpoint is missing config")
    task_spec = _task_spec_from_payload(payload.get("task"))
    domain_payload = payload.get("domain_adaptation")
    domain_config = domain_payload if isinstance(domain_payload, dict) else {}

    common_kwargs = {
        "conv_channels": int(config["conv_channels"]),
        "num_conv_layers": int(config.get("num_conv_layers", 3)),
        "kernel_size": int(config["kernel_size"]),
        "stride": int(config.get("stride", 1)),
        "dilation": int(config.get("dilation", 1)),
        "dropout": float(config["dropout"]),
        "use_residual": bool(config.get("use_residual", True)),
        "normalization": str(config.get("normalization", "layernorm")),
        "pooling": str(config.get("pooling", "mean_max")),
        "include_total_layer_count": bool(config.get("include_total_layer_count", True)),
        "depth_feature_mode": str(config.get("depth_feature_mode", "both")),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "random_state": int(config.get("random_state", 42)),
        "task_spec": task_spec,
        "max_epochs": int(config.get("max_epochs", CNN_MAX_EPOCHS)),
        "batch_size": int(config.get("batch_size", CNN_BATCH_SIZE)),
        "patience": int(config.get("patience", CNN_PATIENCE)),
        "class_weight_loss": bool(config.get("class_weight_loss", False)),
        "rank_label_weight_loss": bool(config.get("rank_label_weight_loss", False)),
    }

    if backend == "cnn_1d_dann":
        lambda_schedule = domain_config.get("lambda_schedule")
        if not isinstance(lambda_schedule, dict):
            lambda_schedule = {}
        model: CNN1DSupervisedModel = CNN1DDANNSupervisedModel(
            **common_kwargs,
            source_rank=int(domain_config.get("source_rank", 256)),
            dann_lambda_max=float(lambda_schedule.get("lambda_max", 1.0)),
            dann_lambda_gamma=float(lambda_schedule.get("gamma", 10.0)),
        )
        domain_class_names = tuple(str(x) for x in domain_config.get("domain_class_names", []))
        domain_rank_values = tuple(int(x) for x in domain_config.get("domain_rank_values", []))
        if not domain_class_names:
            domain_output_dim = 0
            state_dict = payload.get("state_dict")
            if isinstance(state_dict, dict):
                domain_weight = state_dict.get("domain_head.weight")
                if domain_weight is not None and hasattr(domain_weight, "shape"):
                    domain_output_dim = int(domain_weight.shape[0])
            domain_class_names = tuple(f"domain_{idx}" for idx in range(domain_output_dim))
        if not domain_rank_values:
            domain_rank_values = tuple(range(len(domain_class_names)))
        dann_model = model
        assert isinstance(dann_model, CNN1DDANNSupervisedModel)
        dann_model.domain_classes_ = np.arange(len(domain_class_names), dtype=np.int64)
        dann_model.domain_class_names_ = domain_class_names
        dann_model.domain_rank_values_ = domain_rank_values
        dann_model.domain_class_to_index_ = {
            name: int(idx) for idx, name in enumerate(domain_class_names)
        }
    else:
        model = CNN1DSupervisedModel(**common_kwargs)

    channel_layout = _channel_layout_from_payload(payload.get("channel_layout"))
    normalization_payload = payload.get("normalization")
    if not isinstance(normalization_payload, dict):
        raise ValueError("CNN checkpoint is missing normalization")
    channel_mean = np.asarray(normalization_payload.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(normalization_payload.get("channel_std"), dtype=np.float32)
    if channel_mean.ndim != 1 or channel_std.ndim != 1 or channel_mean.shape != channel_std.shape:
        raise ValueError("CNN checkpoint normalization arrays must be aligned 1D arrays")

    model.channel_layout_ = channel_layout
    model.normalization_stats_ = CNNNormalizationStats(
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    model.channel_mean_ = channel_mean
    model.channel_std_ = channel_std
    model.input_channels_ = int(config.get("input_channels") or channel_layout.input_dim)
    model.classes_ = np.asarray(payload.get("classes", np.arange(task_spec.n_classes)), dtype=np.int32)
    model.class_names_ = tuple(str(x) for x in payload.get("class_names", task_spec.class_names))
    fit_summary = payload.get("fit_summary")
    model._fit_summary = dict(fit_summary) if isinstance(fit_summary, dict) else {}

    model.model_ = model._build_model(input_channels=int(model.input_channels_ or channel_layout.input_dim))
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("CNN checkpoint is missing state_dict")
    model.model_.load_state_dict(state_dict)
    model.model_.eval()
    return model

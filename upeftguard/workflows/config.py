"""Shared defaults and path/configuration helpers for experiment stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..features.spectral import DEFAULT_SPECTRAL_FEATURES
from ..supervised.contracts import TABULAR_SPECTRAL_REPRESENTATION_KIND
from ..supervised.models.registry import supported_representation_kinds
from ..utilities.core.run_context import create_run_context
from .experiment import ExperimentContext, create_experiment_context


DEFAULT_EXPERIMENT_FEATURES = list(DEFAULT_SPECTRAL_FEATURES)
DEFAULT_EXPERIMENT_AGGREGATION_LAYOUT = "layer_sequence"


def coerce_features(features: list[str] | tuple[str, ...] | None) -> list[str]:
    if features is None:
        return list(DEFAULT_EXPERIMENT_FEATURES)
    return [str(feature) for feature in features]


def resolve_split_by_folder(split_by_folder: bool | None, calibration_split: int | None) -> bool:
    if split_by_folder is None:
        return calibration_split is not None
    return bool(split_by_folder)


def model_requires_feature_aggregation(model_name: str) -> bool:
    if str(model_name) == "all":
        return True
    return TABULAR_SPECTRAL_REPRESENTATION_KIND not in supported_representation_kinds(str(model_name))


def feature_params(
    *,
    features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
) -> dict[str, Any]:
    return {
        "block_size": int(stream_block_size),
        "dtype": str(dtype_name),
        "spectral_features": list(features),
        "spectral_sv_top_k": int(spectral_sv_top_k),
        "spectral_moment_source": str(spectral_moment_source),
        "spectral_qv_sum_mode": str(spectral_qv_sum_mode),
        "spectral_entrywise_delta_mode": str(spectral_entrywise_delta_mode),
        "spectral_attention_granularity": str(spectral_attention_granularity),
    }


def stage_context(
    output_root: Path,
    *,
    run_id: str | None,
    workflow: str,
    stage: str,
    backend: str,
) -> tuple[ExperimentContext, Any]:
    experiment = create_experiment_context(
        output_root=output_root,
        run_id=run_id,
        workflow=workflow,
        backend=backend,
    )
    stage_directories = {
        "extraction": ("features",),
        "aggregation": ("features",),
        "training": ("models", "reports"),
        "inference": ("reports",),
        "explanation": ("reports",),
    }[stage]
    if backend == "slurm":
        stage_directories = (*stage_directories, "logs")
    context = create_run_context(
        pipeline=stage,
        output_root=experiment.output_root,
        run_id=experiment.run_id,
        run_dir=experiment.stage_dir(stage),
        create_directories=stage_directories,
    )
    return experiment, context

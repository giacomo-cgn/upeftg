"""Input alignment and cohort selection for grouped explanations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from ...artifacts.metadata.spectral import load_spectral_metadata
from ...artifacts.paths import resolve_existing_companion_path
from ...artifacts.provenance.datasets import load_dataset_reference_report
from ...contracts.spectral import feature_block_name
from ...features.spectral import (
    build_spectral_feature_names,
    resolve_spectral_attention_granularity,
    resolve_spectral_features,
    resolve_spectral_moment_source,
    resolve_spectral_qv_sum_mode,
)
from ..data.bundles import resolve_supervised_feature_bundle_paths


SUPPORTED_BACKGROUND_LABELS = ("all", "clean", "backdoor")


@dataclass(frozen=True)
class ExplanationSample:
    row_index: int
    model_name: str
    label: int
    attack_name: str


@dataclass(frozen=True)
class PreparedExplanationData:
    feature_path: Path
    source_feature_shape: tuple[int, int]
    model_feature_names: tuple[str, ...]
    model_column_indices: np.ndarray
    background_samples: tuple[ExplanationSample, ...]
    explanation_samples: tuple[ExplanationSample, ...]
    source_metadata_path: Path
    dataset_reference_path: Path


def _unique_index(names: list[str], *, context: str) -> dict[str, int]:
    result: dict[str, int] = {}
    duplicates: list[str] = []
    for index, name in enumerate(names):
        if name in result:
            duplicates.append(name)
        else:
            result[name] = index
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate model names in {context}. Examples: {preview}")
    return result


def _ordered_block_names(feature_names: list[str]) -> list[str]:
    blocks: list[str] = []
    seen: set[str] = set()
    for name in feature_names:
        block = feature_block_name(name)
        if block not in seen:
            seen.add(block)
            blocks.append(block)
    return blocks


def model_column_selection(
    *,
    source_feature_names: list[str],
    source_width: int,
    feature_configuration: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Resolve the raw bundle columns used by the fitted checkpoint, in training order."""

    if len(source_feature_names) != int(source_width):
        raise ValueError(
            f"Spectral metadata contains {len(source_feature_names)} feature names, "
            f"but the raw matrix has width {source_width}"
        )
    selected_families = resolve_spectral_features(feature_configuration.get("spectral_features"))
    sv_top_k = int(feature_configuration.get("spectral_sv_top_k", 8))
    moment_source = resolve_spectral_moment_source(feature_configuration.get("spectral_moment_source"))
    qv_sum_mode = resolve_spectral_qv_sum_mode(feature_configuration.get("spectral_qv_sum_mode"))
    resolve_spectral_attention_granularity(feature_configuration.get("spectral_attention_granularity"))

    all_blocks = _ordered_block_names(source_feature_names)
    if qv_sum_mode == "none":
        selected_blocks = [name for name in all_blocks if ".qv_sum" not in name]
    elif qv_sum_mode == "only":
        selected_blocks = [name for name in all_blocks if ".qv_sum" in name]
    else:
        selected_blocks = all_blocks
    if not selected_blocks:
        raise ValueError(f"Raw feature bundle has no blocks compatible with spectral_qv_sum_mode={qv_sum_mode}")

    expected_names = build_spectral_feature_names(
        block_names=selected_blocks,
        selected_features=selected_families,
        sv_top_k=sv_top_k,
        spectral_moment_source=moment_source,
        shorten_block_names=False,
    )
    source_index = _unique_index(source_feature_names, context="spectral feature metadata")
    missing = [name for name in expected_names if name not in source_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Raw feature bundle is missing columns required by the checkpoint. Examples: {preview}")
    return np.asarray([source_index[name] for name in expected_names], dtype=np.int64), expected_names


def _load_selection_names(path: Path) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Selection file not found: {resolved}")
    if resolved.suffix.lower() == ".csv":
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "model_name" not in reader.fieldnames:
                raise ValueError(f"Selection CSV must contain a model_name column: {resolved}")
            names = [str(row.get("model_name", "")).strip() for row in reader]
    elif resolved.suffix.lower() == ".json":
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("model_names", payload.get("samples"))
        if not isinstance(payload, list):
            raise ValueError(f"Selection JSON must be a list or contain model_names/samples: {resolved}")
        names = [
            str(item.get("model_name", "")).strip() if isinstance(item, dict) else str(item).strip() for item in payload
        ]
    else:
        raise ValueError(f"Selection file must be .csv or .json: {resolved}")
    if not names or any(not name for name in names):
        raise ValueError(f"Selection file contains no rows or an empty model_name: {resolved}")
    _unique_index(names, context=str(resolved))
    return names


def _sample_stratified(
    samples: list[ExplanationSample],
    *,
    max_samples: int | None,
    random_state: int,
    stratum: Callable[[ExplanationSample], tuple[Any, ...]],
) -> list[ExplanationSample]:
    if max_samples is None or int(max_samples) >= len(samples):
        return list(samples)
    limit = int(max_samples)
    if limit <= 0:
        raise ValueError("Sample limits must be positive")

    by_stratum: dict[tuple[Any, ...], list[ExplanationSample]] = {}
    for sample in samples:
        by_stratum.setdefault(stratum(sample), []).append(sample)
    keys = sorted(by_stratum, key=lambda value: tuple(str(part) for part in value))
    exact = {key: limit * len(by_stratum[key]) / len(samples) for key in keys}
    allocation = {key: min(len(by_stratum[key]), int(np.floor(exact[key]))) for key in keys}
    remaining = limit - sum(allocation.values())
    priority = sorted(keys, key=lambda key: (-(exact[key] - allocation[key]), tuple(str(x) for x in key)))
    while remaining:
        progressed = False
        for key in priority:
            if allocation[key] < len(by_stratum[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    rng = np.random.default_rng(int(random_state))
    selected: list[ExplanationSample] = []
    for key in keys:
        members = by_stratum[key]
        indices = rng.choice(len(members), size=allocation[key], replace=False)
        selected.extend(members[int(index)] for index in sorted(indices.tolist()))
    return sorted(selected, key=lambda sample: sample.row_index)


def _samples_from_names(
    names: list[str],
    *,
    row_index: dict[str, int],
    provenance_index: dict[str, Any],
) -> list[ExplanationSample]:
    missing_rows = [name for name in names if name not in row_index]
    if missing_rows:
        preview = ", ".join(missing_rows[:5])
        raise ValueError(f"Selection contains model names absent from the raw bundle. Examples: {preview}")
    missing_provenance = [name for name in names if name not in provenance_index]
    if missing_provenance:
        preview = ", ".join(missing_provenance[:5])
        raise ValueError(f"Selection contains model names without label/attack provenance. Examples: {preview}")

    samples: list[ExplanationSample] = []
    for name in names:
        entry = provenance_index[name]
        if not isinstance(entry, dict) or entry.get("label") not in {0, 1, "0", "1"}:
            raise ValueError(f"Model {name!r} does not have a binary label in dataset provenance")
        label = int(entry["label"])
        attack_name = str(entry.get("attack_name") or "unknown").strip() or "unknown"
        if label == 1 and attack_name.lower() == "unknown":
            raise ValueError(f"Positive model {name!r} is missing attack_name provenance")
        samples.append(
            ExplanationSample(
                row_index=int(row_index[name]),
                model_name=name,
                label=label,
                attack_name=attack_name,
            )
        )
    return samples


def prepare_explanation_data(
    *,
    feature_file: Path,
    background_selection: Path,
    explanation_selection: Path,
    inference_contract: dict[str, Any],
    expected_model_width: int,
    background_label: str = "all",
    background_max_samples: int | None = None,
    explanation_max_samples: int | None = None,
    random_state: int = 42,
) -> PreparedExplanationData:
    paths = resolve_supervised_feature_bundle_paths(Path(feature_file))
    features = np.load(paths.feature_path, mmap_mode="r")
    if features.ndim != 2:
        raise ValueError(f"Grouped linear SHAP requires a raw 2D feature matrix, got shape={features.shape}")
    if paths.metadata_path is None:
        raise FileNotFoundError(f"Raw feature bundle is missing spectral metadata for {paths.feature_path}")
    metadata = load_spectral_metadata(paths.metadata_path)
    source_names = metadata.get("feature_names")
    if not isinstance(source_names, list) or not source_names:
        raise ValueError("Raw feature metadata must include feature_names")

    with paths.model_names_path.open("r", encoding="utf-8") as handle:
        model_names = [str(value) for value in json.load(handle)]
    if len(model_names) != int(features.shape[0]):
        raise ValueError("Raw feature row count does not match its model_names companion")
    row_index = _unique_index(model_names, context=str(paths.model_names_path))

    reference_path = paths.feature_path.parent / "dataset_reference_report.json"
    reference = load_dataset_reference_report(reference_path)
    provenance_index = reference.get("model_index")
    if not isinstance(provenance_index, dict):
        raise ValueError(f"Dataset reference state is missing model_index: {reference_path}")

    label_path = resolve_existing_companion_path(paths.feature_path, "_labels.npy", required=True)
    bundle_labels = np.asarray(np.load(label_path, mmap_mode="r")).reshape(-1)
    if bundle_labels.size != len(model_names):
        raise ValueError("Raw feature row count does not match its labels companion")
    for name, index in row_index.items():
        entry = provenance_index.get(name)
        if isinstance(entry, dict) and entry.get("label") is not None:
            if int(entry["label"]) != int(bundle_labels[index]):
                raise ValueError(f"Label mismatch between feature bundle and provenance for model {name!r}")

    background_names = _load_selection_names(background_selection)
    explanation_names = _load_selection_names(explanation_selection)
    overlap = sorted(set(background_names) & set(explanation_names))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(
            f"Background and explanation selections must be non-overlapping. Overlapping examples: {preview}"
        )
    background_samples = _samples_from_names(background_names, row_index=row_index, provenance_index=provenance_index)
    explanation_samples = _samples_from_names(explanation_names, row_index=row_index, provenance_index=provenance_index)

    resolved_background_label = str(background_label).strip().lower()
    if resolved_background_label not in SUPPORTED_BACKGROUND_LABELS:
        raise ValueError(
            f"Unknown background label {background_label!r}; choose from {list(SUPPORTED_BACKGROUND_LABELS)}"
        )
    if resolved_background_label != "all":
        wanted = 0 if resolved_background_label == "clean" else 1
        background_samples = [sample for sample in background_samples if sample.label == wanted]
    if not background_samples:
        raise ValueError(f"Background selection has no samples after background_label={resolved_background_label}")

    background_samples = _sample_stratified(
        background_samples,
        max_samples=background_max_samples,
        random_state=random_state,
        stratum=lambda sample: (sample.label,),
    )
    explanation_samples = _sample_stratified(
        explanation_samples,
        max_samples=explanation_max_samples,
        random_state=random_state,
        stratum=lambda sample: (sample.label, sample.attack_name if sample.label == 1 else "clean"),
    )

    feature_configuration = inference_contract.get("feature_configuration")
    if not isinstance(feature_configuration, dict):
        raise ValueError("Checkpoint inference contract is missing feature_configuration")
    column_indices, selected_names = model_column_selection(
        source_feature_names=[str(value) for value in source_names],
        source_width=int(features.shape[1]),
        feature_configuration=feature_configuration,
    )
    if len(selected_names) != int(expected_model_width):
        raise ValueError(
            f"Checkpoint expects {expected_model_width} input columns, but its feature contract "
            f"selects {len(selected_names)} raw bundle columns"
        )

    return PreparedExplanationData(
        feature_path=paths.feature_path,
        source_feature_shape=(int(features.shape[0]), int(features.shape[1])),
        model_feature_names=tuple(selected_names),
        model_column_indices=column_indices,
        background_samples=tuple(background_samples),
        explanation_samples=tuple(explanation_samples),
        source_metadata_path=paths.metadata_path,
        dataset_reference_path=reference_path,
    )


def raw_feature_batches(
    prepared: PreparedExplanationData,
    samples: tuple[ExplanationSample, ...],
    *,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Yield selected rows and model columns without loading the complete raw bundle."""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("batch_size must be positive")
    features = np.load(prepared.feature_path, mmap_mode="r")
    columns = prepared.model_column_indices
    for start in range(0, len(samples), size):
        row_indices = [sample.row_index for sample in samples[start : start + size]]
        rows = np.asarray(features[row_indices])
        yield np.asarray(rows[:, columns], dtype=np.float32)

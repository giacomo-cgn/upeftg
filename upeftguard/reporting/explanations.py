"""Writers for grouped SHAP local values and global cohort summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..supervised.explainability.data import ExplanationSample
from ..supervised.explainability.grouping import GroupSchema
from ..utilities.core.serialization import json_ready


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)
        handle.write("\n")
    return path


def _attack_filename(attack_name: str) -> str:
    value = _SAFE_FILENAME_RE.sub("_", str(attack_name).strip()).strip("._")
    return value or "unknown"


def _write_sample_selection(path: Path, samples: tuple[ExplanationSample, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "model_name", "label", "attack_name"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "row_index": int(sample.row_index),
                    "model_name": sample.model_name,
                    "label": int(sample.label),
                    "attack_name": sample.attack_name,
                }
            )
    return path


def _write_global_importance(
    path: Path,
    *,
    values: np.ndarray,
    schema: GroupSchema,
) -> Path:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or int(matrix.shape[1]) != schema.n_groups:
        raise ValueError(f"Grouped SHAP matrix shape {matrix.shape} does not match {schema.n_groups} groups")
    path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.shape[0]:
        mean_abs = np.mean(np.abs(matrix), axis=0)
        mean_signed = np.mean(matrix, axis=0)
        std_signed = np.std(matrix, axis=0)
        order = np.argsort(-mean_abs, kind="stable")
        ranks = np.empty(schema.n_groups, dtype=np.int64)
        ranks[order] = np.arange(1, schema.n_groups + 1, dtype=np.int64)
    else:
        mean_abs = np.full(schema.n_groups, np.nan)
        mean_signed = np.full(schema.n_groups, np.nan)
        std_signed = np.full(schema.n_groups, np.nan)
        ranks = np.zeros(schema.n_groups, dtype=np.int64)

    fields = [
        "rank",
        "group_index",
        "group_name",
        *schema.axes,
        "column_count",
        "n_samples",
        "mean_abs_shap",
        "mean_shap",
        "std_shap",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        indices = range(schema.n_groups) if not matrix.shape[0] else order.tolist()
        for index in indices:
            row = {
                "rank": int(ranks[index]) if matrix.shape[0] else "",
                "group_index": int(index),
                "group_name": schema.group_names[index],
                "column_count": int(schema.group_column_counts[index]),
                "n_samples": int(matrix.shape[0]),
                "mean_abs_shap": float(mean_abs[index]),
                "mean_shap": float(mean_signed[index]),
                "std_shap": float(std_signed[index]),
            }
            row.update({axis: schema.group_value(index, axis) for axis in schema.axes})
            writer.writerow(row)
    return path


def _cohort_masks(
    samples: tuple[ExplanationSample, ...],
) -> list[tuple[str, np.ndarray]]:
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    attacks = np.asarray([sample.attack_name for sample in samples], dtype=str)
    masks: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(samples), dtype=bool)),
        ("clean", labels == 0),
        ("backdoor", labels == 1),
    ]
    for attack_name in sorted(set(attacks[labels == 1].tolist())):
        masks.append(
            (
                f"attack_{_attack_filename(attack_name)}",
                (labels == 1) & (attacks == attack_name),
            )
        )
    return masks


def _write_grouping_outputs(
    *,
    run_dir: Path,
    reports_dir: Path,
    grouped_values: np.ndarray,
    base_log_odds: np.ndarray,
    predicted_log_odds: np.ndarray,
    background_mean: np.ndarray,
    samples: tuple[ExplanationSample, ...],
    schema: GroupSchema,
    plot_max_display: int,
    max_local_plots: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    values = np.asarray(grouped_values, dtype=np.float64)
    if values.shape != (len(samples), schema.n_groups):
        raise ValueError(f"Grouped SHAP values have shape {values.shape}; expected {(len(samples), schema.n_groups)}")
    base_values = np.asarray(base_log_odds, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted_log_odds, dtype=np.float64).reshape(-1)
    background_mean_values = np.asarray(background_mean, dtype=np.float64).reshape(-1)
    if base_values.size != len(samples) or predicted_values.size != len(samples):
        raise ValueError("Base and predicted log-odds arrays must have one value per explanation sample")
    if background_mean_values.size != schema.n_columns:
        raise ValueError("Background mean must have one value per fitted model column")

    reports_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    attacks = np.asarray([sample.attack_name for sample in samples], dtype=str)
    names = np.asarray([sample.model_name for sample in samples], dtype=str)
    schema_path = _write_json(reports_dir / "group_schema.json", schema.to_dict())
    local_path = reports_dir / "grouped_shap_values.npz"
    np.savez_compressed(
        local_path,
        grouped_shap_values=values,
        model_names=names,
        labels=labels,
        attack_names=attacks,
        base_log_odds=base_values,
        predicted_log_odds=predicted_values,
        background_mean_model_space=background_mean_values,
        grouping_axes=np.asarray(schema.axes, dtype=str),
        axis_values=np.asarray(
            [json.dumps(list(axis_values)) for axis_values in schema.axis_values],
            dtype=str,
        ),
        group_names=np.asarray(schema.group_names, dtype=str),
        group_keys=np.asarray(schema.group_keys, dtype=str),
        group_column_counts=np.asarray(schema.group_column_counts, dtype=np.int64),
    )

    filenames: set[str] = set()
    global_paths: dict[str, Path] = {}
    cohort_counts: dict[str, int] = {}
    for cohort_name, mask in _cohort_masks(samples):
        filename = f"global_importance_{cohort_name}.csv"
        if filename.lower() in filenames:
            raise ValueError(f"Attack names produce colliding global-importance filename {filename!r}")
        filenames.add(filename.lower())
        cohort_counts[cohort_name] = int(np.count_nonzero(mask))
        global_paths[cohort_name] = _write_global_importance(
            reports_dir / filename,
            values=values[mask],
            schema=schema,
        )

    from .explanation_plots import write_grouped_explanation_plots

    plot_paths = write_grouped_explanation_plots(
        run_dir=run_dir,
        reports_dir=reports_dir,
        grouped_values=values,
        base_log_odds=base_values,
        predicted_log_odds=predicted_values,
        samples=samples,
        schema=schema,
        max_display=plot_max_display,
        max_local_plots=max_local_plots,
    )
    paths = {
        "group_schema": schema_path,
        "grouped_shap_values": local_path,
        **{f"global_importance_{name}": path for name, path in global_paths.items()},
        **plot_paths,
    }
    summary = {
        "axes": list(schema.axes),
        "n_groups": schema.n_groups,
        "cohort_sample_counts": cohort_counts,
        "grouped_shap_values": str(local_path.relative_to(run_dir)),
        "global_importance_files": {name: str(path.relative_to(run_dir)) for name, path in global_paths.items()},
        "plot_files": {name: str(path.relative_to(run_dir)) for name, path in plot_paths.items()},
    }
    return paths, summary


def write_grouped_explanation_bundle(
    *,
    run_dir: Path,
    grouped_values: np.ndarray,
    base_log_odds: np.ndarray,
    predicted_log_odds: np.ndarray,
    background_mean: np.ndarray,
    background_samples: tuple[ExplanationSample, ...],
    samples: tuple[ExplanationSample, ...],
    schema: GroupSchema,
    explanation_config: dict[str, Any],
    report: dict[str, Any],
    plot_max_display: int = 20,
    max_local_plots: int = 20,
) -> dict[str, Path]:
    """Write local values once and derive every global cohort from that matrix."""

    run_dir = Path(run_dir).expanduser().resolve()
    inputs_dir = run_dir / "inputs"
    reports_dir = run_dir / "reports"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_json(inputs_dir / "explanation_config.json", explanation_config)
    background_samples_path = _write_sample_selection(inputs_dir / "background_samples.csv", background_samples)
    explanation_samples_path = _write_sample_selection(inputs_dir / "explanation_samples.csv", samples)
    grouping_paths, grouping_summary = _write_grouping_outputs(
        run_dir=run_dir,
        reports_dir=reports_dir,
        grouped_values=grouped_values,
        base_log_odds=base_log_odds,
        predicted_log_odds=predicted_log_odds,
        background_mean=background_mean,
        samples=samples,
        schema=schema,
        plot_max_display=plot_max_display,
        max_local_plots=max_local_plots,
    )
    report_payload = {
        "schema_version": 2,
        **dict(report),
        "groupings": {schema.name: grouping_summary},
    }
    report_path = _write_json(reports_dir / "report.json", report_payload)
    return {
        "explanation_config": config_path,
        "background_samples": background_samples_path,
        "explanation_samples": explanation_samples_path,
        "report": report_path,
        **grouping_paths,
    }


def write_multi_grouped_explanation_bundle(
    *,
    run_dir: Path,
    grouped_values: dict[str, np.ndarray],
    base_log_odds: np.ndarray,
    predicted_log_odds: np.ndarray,
    background_mean: np.ndarray,
    background_samples: tuple[ExplanationSample, ...],
    samples: tuple[ExplanationSample, ...],
    schemas: dict[str, GroupSchema],
    explanation_config: dict[str, Any],
    report: dict[str, Any],
    plot_max_display: int = 20,
    max_local_plots: int = 20,
) -> dict[str, Path]:
    """Write strategy-scoped artifacts for multiple aggregations from one SHAP pass."""

    run_dir = Path(run_dir).expanduser().resolve()
    inputs_dir = run_dir / "inputs"
    reports_dir = run_dir / "reports"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if set(grouped_values) != set(schemas):
        raise ValueError(
            "Grouped SHAP result strategies do not match schemas: "
            f"values={sorted(grouped_values)}, schemas={sorted(schemas)}"
        )

    paths: dict[str, Path] = {
        "explanation_config": _write_json(
            inputs_dir / "explanation_config.json",
            explanation_config,
        ),
        "background_samples": _write_sample_selection(
            inputs_dir / "background_samples.csv",
            background_samples,
        ),
        "explanation_samples": _write_sample_selection(
            inputs_dir / "explanation_samples.csv",
            samples,
        ),
    }
    grouping_summaries: dict[str, Any] = {}
    for name, schema in schemas.items():
        grouping_dir = reports_dir / "groupings" / name
        strategy_paths, summary = _write_grouping_outputs(
            run_dir=run_dir,
            reports_dir=grouping_dir,
            grouped_values=grouped_values[name],
            base_log_odds=base_log_odds,
            predicted_log_odds=predicted_log_odds,
            background_mean=background_mean,
            samples=samples,
            schema=schema,
            plot_max_display=plot_max_display,
            max_local_plots=max_local_plots,
        )
        grouping_summaries[name] = summary
        key_prefix = name.replace("+", "_")
        paths.update({f"{key_prefix}_{key}": path for key, path in strategy_paths.items()})

    report_payload = {
        "schema_version": 2,
        **dict(report),
        "groupings": grouping_summaries,
    }
    paths["report"] = _write_json(reports_dir / "report.json", report_payload)
    return paths

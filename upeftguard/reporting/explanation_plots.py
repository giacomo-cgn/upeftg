"""Static plots for grouped linear SHAP explanation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

import matplotlib
import numpy as np

from ..supervised.explainability.data import ExplanationSample
from ..supervised.explainability.grouping import GroupSchema


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(value: str) -> str:
    resolved = _SAFE_FILENAME_RE.sub("_", str(value).strip()).strip("._")
    return resolved or "unknown"


def _cohort_masks(samples: tuple[ExplanationSample, ...]) -> list[tuple[str, str, np.ndarray]]:
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    attacks = np.asarray([sample.attack_name for sample in samples], dtype=str)
    cohorts: list[tuple[str, str, np.ndarray]] = [
        ("all", "All explanation samples", np.ones(len(samples), dtype=bool)),
        ("clean", "Clean", labels == 0),
        ("backdoor", "Backdoored", labels == 1),
    ]
    for attack_name in sorted(set(attacks[labels == 1].tolist())):
        cohorts.append(
            (
                f"attack_{_safe_filename(attack_name)}",
                f"Attack: {attack_name}",
                (labels == 1) & (attacks == attack_name),
            )
        )
    return cohorts


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close("all")
    return path


def _shap_explanation(
    *,
    shap_module: Any,
    values: np.ndarray,
    base_values: np.ndarray,
    feature_names: tuple[str, ...],
) -> Any:
    value_array = np.asarray(values, dtype=np.float64)
    resolved_base_values: float | np.ndarray
    if value_array.ndim == 1:
        resolved_base_values = float(np.asarray(base_values, dtype=np.float64).reshape(-1)[0])
    else:
        resolved_base_values = np.asarray(base_values, dtype=np.float64)
    return shap_module.Explanation(
        values=value_array,
        base_values=resolved_base_values,
        feature_names=list(feature_names),
    )


def _write_shap_bar(
    path: Path,
    *,
    explanation: Any,
    title: str,
    max_display: int,
    shap_module: Any,
) -> Path:
    plt.close("all")
    shap_module.plots.bar(explanation, max_display=max_display, show=False)
    plt.gca().set_title(title)
    return _save_current_figure(path)


def _write_beeswarm(
    path: Path,
    *,
    explanation: Any,
    title: str,
    max_display: int,
    shap_module: Any,
) -> Path:
    plt.close("all")
    shap_module.plots.beeswarm(
        explanation,
        max_display=max_display,
        color="#1f77b4",
        color_bar=False,
        show=False,
    )
    plt.gca().set_title(title)
    return _save_current_figure(path)


def _write_shap_heatmap(
    path: Path,
    *,
    explanation: Any,
    title: str,
    max_display: int,
    shap_module: Any,
) -> Path:
    plt.close("all")
    shap_module.plots.heatmap(explanation, max_display=max_display, show=False)
    plt.gca().set_title(title)
    return _save_current_figure(path)


def _write_two_axis_heatmap(
    path: Path,
    *,
    matrix: np.ndarray,
    schema: GroupSchema,
    title: str,
    colorbar_label: str,
    diverging: bool = False,
) -> Path:
    if schema.n_dimensions != 2:
        raise ValueError(f"Two-axis heatmap requires two grouping dimensions, got {schema.axes}")
    row_axis, column_axis = schema.axes
    row_values, column_values = schema.axis_values
    values = np.asarray(matrix, dtype=np.float64).reshape(len(row_values), len(column_values))
    width = max(10.0, 0.9 * len(column_values))
    height = max(6.0, 0.34 * len(row_values))
    fig, ax = plt.subplots(figsize=(width, height))
    image_kwargs: dict[str, Any] = {
        "aspect": "auto",
        "interpolation": "nearest",
        "cmap": "RdBu_r" if diverging else "viridis",
    }
    if diverging:
        limit = float(np.max(np.abs(values), initial=0.0))
        if limit == 0.0:
            limit = 1.0
        image_kwargs.update({"vmin": -limit, "vmax": limit})
    image = ax.imshow(values, **image_kwargs)
    ax.set_xticks(np.arange(len(column_values)))
    ax.set_xticklabels(column_values, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_values)))
    ax.set_yticklabels(row_values)
    ax.set_xlabel(column_axis.replace("_", " ").title())
    ax.set_ylabel(row_axis.replace("_", " ").title())
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _write_cohort_comparison(
    path: Path,
    *,
    values: np.ndarray,
    cohorts: list[tuple[str, str, np.ndarray]],
    schema: GroupSchema,
    max_display: int,
) -> Path | None:
    nonempty = [(key, label, mask) for key, label, mask in cohorts if np.any(mask)]
    if not nonempty:
        return None
    global_importance = np.mean(np.abs(values), axis=0)
    top_indices = np.argsort(-global_importance, kind="stable")[:max_display]
    cohort_values = np.stack([np.mean(np.abs(values[mask]), axis=0)[top_indices] for _key, _label, mask in nonempty])
    labels = [label for _key, label, _mask in nonempty]
    group_names = [schema.group_names[index] for index in top_indices]

    fig_width = max(12.0, 0.65 * len(top_indices))
    fig_height = max(4.5, 0.55 * len(nonempty))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cohort_values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(np.arange(len(group_names)))
    ax.set_xticklabels(group_names, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Top coalition groups")
    ax.set_title("Mean absolute grouped SHAP by cohort")
    fig.colorbar(image, ax=ax, label="Mean |SHAP| (log-odds)", shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _select_local_indices(
    *,
    values: np.ndarray,
    predicted_log_odds: np.ndarray,
    samples: tuple[ExplanationSample, ...],
    limit: int,
) -> list[int]:
    if limit <= 0:
        return []
    magnitude = np.sum(np.abs(values), axis=1)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    attacks = np.asarray([sample.attack_name for sample in samples], dtype=str)
    selected: list[int] = []

    representative_masks: list[np.ndarray] = [labels == 0, labels == 1]
    representative_masks.extend((labels == 1) & (attacks == attack) for attack in sorted(set(attacks[labels == 1])))
    for mask in representative_masks:
        candidates = np.flatnonzero(mask)
        if not candidates.size:
            continue
        index = int(candidates[np.argmax(magnitude[candidates])])
        if index not in selected:
            selected.append(index)
        if len(selected) == limit:
            return selected

    order = np.argsort(predicted_log_odds, kind="stable")
    left = 0
    right = len(order) - 1
    while len(selected) < limit and left <= right:
        for index in (int(order[right]), int(order[left])):
            if index not in selected:
                selected.append(index)
                if len(selected) == limit:
                    break
        left += 1
        right -= 1
    return selected


def _write_local_selection(
    path: Path,
    *,
    selected_indices: list[int],
    samples: tuple[ExplanationSample, ...],
    predicted_log_odds: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["explanation_index", "model_name", "label", "attack_name", "predicted_log_odds"],
        )
        writer.writeheader()
        for index in selected_indices:
            sample = samples[index]
            writer.writerow(
                {
                    "explanation_index": index,
                    "model_name": sample.model_name,
                    "label": sample.label,
                    "attack_name": sample.attack_name,
                    "predicted_log_odds": float(predicted_log_odds[index]),
                }
            )
    return path


def write_grouped_explanation_plots(
    *,
    run_dir: Path,
    grouped_values: np.ndarray,
    base_log_odds: np.ndarray,
    predicted_log_odds: np.ndarray,
    samples: tuple[ExplanationSample, ...],
    schema: GroupSchema,
    max_display: int = 20,
    max_local_plots: int = 20,
    reports_dir: Path | None = None,
) -> dict[str, Path]:
    """Create global, cohort-comparison, and bounded local SHAP plots."""

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency validated by explanation calculation
        raise RuntimeError("SHAP is required to create grouped explanation plots") from exc

    display_count = int(max_display)
    local_limit = int(max_local_plots)
    if display_count <= 0:
        raise ValueError("plot max_display must be positive")
    if local_limit < 0:
        raise ValueError("max_local_plots must be non-negative")

    values = np.asarray(grouped_values, dtype=np.float64)
    base_values = np.asarray(base_log_odds, dtype=np.float64).reshape(-1)
    logits = np.asarray(predicted_log_odds, dtype=np.float64).reshape(-1)
    if values.shape != (len(samples), schema.n_groups):
        raise ValueError("Grouped SHAP plot inputs do not match the sample/group schema")
    if base_values.size != len(samples) or logits.size != len(samples):
        raise ValueError("Grouped SHAP plot inputs require one base value and log-odds value per sample")

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    resolved_reports_dir = (
        Path(reports_dir).expanduser().resolve() if reports_dir is not None else resolved_run_dir / "reports"
    )
    plot_dir = resolved_reports_dir / "plots"
    global_dir = plot_dir / "global"
    local_dir = plot_dir / "local"
    cohorts = _cohort_masks(samples)
    paths: dict[str, Path] = {}

    for cohort_key, cohort_label, mask in cohorts:
        if not np.any(mask):
            continue
        explanation = _shap_explanation(
            shap_module=shap,
            values=values[mask],
            base_values=base_values[mask],
            feature_names=schema.group_names,
        )
        paths[f"plot_bar_{cohort_key}"] = _write_shap_bar(
            global_dir / f"bar_{cohort_key}.png",
            explanation=explanation,
            title=f"Grouped SHAP importance — {cohort_label}",
            max_display=display_count,
            shap_module=shap,
        )
        paths[f"plot_beeswarm_{cohort_key}"] = _write_beeswarm(
            global_dir / f"beeswarm_{cohort_key}.png",
            explanation=explanation,
            title=f"Grouped SHAP distribution — {cohort_label}",
            max_display=display_count,
            shap_module=shap,
        )
        if schema.n_dimensions == 2:
            paths[f"plot_coalition_heatmap_{cohort_key}"] = _write_two_axis_heatmap(
                global_dir / f"coalition_importance_{cohort_key}.png",
                matrix=np.mean(np.abs(values[mask]), axis=0),
                schema=schema,
                title=f"{schema.name} importance — {cohort_label}",
                colorbar_label="Mean |SHAP| (log-odds)",
            )

    if schema.n_dimensions <= 2:
        all_explanation = _shap_explanation(
            shap_module=shap,
            values=values,
            base_values=base_values,
            feature_names=schema.group_names,
        )
        paths["plot_shap_heatmap_all"] = _write_shap_heatmap(
            global_dir / "shap_heatmap_all.png",
            explanation=all_explanation,
            title="Grouped SHAP values across explanation samples",
            max_display=display_count,
            shap_module=shap,
        )
        comparison = _write_cohort_comparison(
            global_dir / "cohort_importance_comparison.png",
            values=values,
            cohorts=cohorts,
            schema=schema,
            max_display=display_count,
        )
        if comparison is not None:
            paths["plot_cohort_importance_comparison"] = comparison

    clean_mask = np.asarray([sample.label == 0 for sample in samples], dtype=bool)
    backdoor_mask = ~clean_mask
    if np.any(clean_mask) and np.any(backdoor_mask):
        clean_mean = np.mean(values[clean_mask], axis=0)
        backdoor_mean = np.mean(values[backdoor_mask], axis=0)
        if schema.n_dimensions == 2:
            paths["plot_difference_backdoor_minus_clean"] = _write_two_axis_heatmap(
                global_dir / "coalition_difference_backdoor_minus_clean.png",
                matrix=backdoor_mean - clean_mean,
                schema=schema,
                title="Mean SHAP difference: backdoor − clean",
                colorbar_label="Δ mean SHAP (log-odds)",
                diverging=True,
            )
        if (
            schema.n_dimensions <= 2
            and int(np.count_nonzero(clean_mask)) >= 10
            and int(np.count_nonzero(backdoor_mask)) >= 10
        ):
            plt.close("all")
            random_state = np.random.get_state()
            try:
                np.random.seed(0)
                shap.plots.group_difference(
                    values,
                    backdoor_mask,
                    feature_names=list(schema.group_names),
                    max_display=display_count,
                    show=False,
                )
            finally:
                np.random.set_state(random_state)
            plt.gca().set_title("Grouped SHAP difference: backdoor − clean")
            paths["plot_group_difference_backdoor_clean"] = _save_current_figure(
                global_dir / "group_difference_backdoor_minus_clean.png"
            )
        if schema.n_dimensions == 2:
            attack_names = sorted({sample.attack_name for sample in samples if sample.label == 1})
            attacks = np.asarray([sample.attack_name for sample in samples], dtype=str)
            for attack_name in attack_names:
                attack_mask = backdoor_mask & (attacks == attack_name)
                if not np.any(attack_mask):
                    continue
                safe_attack = _safe_filename(attack_name)
                paths[f"plot_difference_attack_{safe_attack}_minus_clean"] = _write_two_axis_heatmap(
                    global_dir / f"coalition_difference_attack_{safe_attack}_minus_clean.png",
                    matrix=np.mean(values[attack_mask], axis=0) - clean_mean,
                    schema=schema,
                    title=f"Mean SHAP difference: {attack_name} − clean",
                    colorbar_label="Δ mean SHAP (log-odds)",
                    diverging=True,
                )

    if schema.n_dimensions <= 2:
        selected_indices = _select_local_indices(
            values=values,
            predicted_log_odds=logits,
            samples=samples,
            limit=min(local_limit, len(samples)),
        )
        paths["plot_local_selection"] = _write_local_selection(
            local_dir / "selection.csv",
            selected_indices=selected_indices,
            samples=samples,
            predicted_log_odds=logits,
        )
        for index in selected_indices:
            sample = samples[index]
            local_explanation = _shap_explanation(
                shap_module=shap,
                values=values[index],
                base_values=base_values[index],
                feature_names=schema.group_names,
            )
            plt.close("all")
            shap.plots.waterfall(local_explanation, max_display=display_count, show=False)
            plt.gca().set_title(
                f"{sample.model_name}\nlabel={sample.label}, attack={sample.attack_name}, log-odds={logits[index]:.3f}"
            )
            filename = f"waterfall_{index:04d}_{_safe_filename(sample.model_name)}.png"
            paths[f"plot_waterfall_{index:04d}"] = _save_current_figure(local_dir / filename)

        if selected_indices:
            plt.close("all")
            shap.plots.decision(
                float(base_values[selected_indices[0]]),
                values[selected_indices],
                feature_names=list(schema.group_names),
                feature_order="importance",
                feature_display_range=slice(-1, -display_count - 1, -1),
                link="logit",
                color_bar=True,
                title="Selected grouped-SHAP decision paths",
                show=False,
                ignore_warnings=True,
            )
            paths["plot_decision_selected"] = _save_current_figure(local_dir / "decision_selected_probability.png")

    manifest_path = plot_dir / "plot_manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "plot_max_display": display_count,
        "max_local_plots": local_limit,
        "grouping": schema.name,
        "files": {name: str(path.relative_to(resolved_run_dir)) for name, path in paths.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
        handle.write("\n")
    paths["plot_manifest"] = manifest_path
    return paths

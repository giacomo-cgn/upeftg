"""Lifecycle entry point for grouped linear SHAP explanations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ...features.spectral import resolve_spectral_features
from ...reporting.explanations import write_multi_grouped_explanation_bundle
from ...utilities.core.run_context import create_run_context
from ...utilities.core.experiment import experiment_context_from_stage_dir
from ..explainability.data import prepare_explanation_data, raw_feature_batches
from ..explainability.grouping import (
    build_group_schemas,
    resolve_grouping_strategies,
)
from ..explainability.linear import (
    explain_linear_log_odds_multi,
    linear_pipeline_parts,
    transformed_background_mean,
)
from ..models.artifact import load_supervised_artifact
from .timing import append_stage_timing


def run_linear_grouped_explanation(
    *,
    checkpoint: Path,
    feature_file: Path,
    background_selection: Path,
    explanation_selection: Path,
    output_root: Path,
    run_id: str | None,
    output_run_dir: Path | None = None,
    background_label: str = "all",
    background_max_samples: int | None = None,
    explanation_max_samples: int | None = None,
    random_state: int = 42,
    batch_size: int = 128,
    grouping: list[str] | tuple[str, ...] | None = None,
    plot_max_display: int = 20,
    max_local_plots: int = 20,
) -> dict[str, Any]:
    """Explain a fitted binary logistic model and write local/global artifacts."""

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    checkpoint = Path(checkpoint).expanduser().resolve()
    feature_file = Path(feature_file).expanduser().resolve()
    background_selection = Path(background_selection).expanduser().resolve()
    explanation_selection = Path(explanation_selection).expanduser().resolve()

    artifact = load_supervised_artifact(checkpoint)
    contract = artifact.inference_contract
    if str(contract.get("model_name")) != "logistic_regression":
        raise ValueError(
            "Grouped linear SHAP currently requires a logistic_regression checkpoint; "
            f"got {contract.get('model_name')!r}"
        )
    task = contract.get("task")
    if not isinstance(task, dict) or task.get("task_mode") != "binary":
        raise ValueError("Grouped linear SHAP currently supports binary checkpoints only")
    if task.get("class_to_index") != {"clean": 0, "backdoored": 1}:
        raise ValueError("Expected checkpoint classes clean=0 and backdoored=1 for log-odds interpretation")
    input_normalization = contract.get("input_normalization")
    if not isinstance(input_normalization, dict) or str(input_normalization.get("mode", "none")) != "none":
        raise ValueError(
            "Grouped linear SHAP currently requires input_normalization.mode=none; "
            "the fitted pipeline's own normalizer is handled directly"
        )

    normalizer, estimator = linear_pipeline_parts(artifact.model)
    expected_width = int(getattr(normalizer, "n_features_in_", estimator.coef_.shape[1]))
    prepared = prepare_explanation_data(
        feature_file=feature_file,
        background_selection=background_selection,
        explanation_selection=explanation_selection,
        inference_contract=contract,
        expected_model_width=expected_width,
        background_label=background_label,
        background_max_samples=background_max_samples,
        explanation_max_samples=explanation_max_samples,
        random_state=random_state,
    )
    feature_config = dict(contract["feature_configuration"])
    family_order = resolve_spectral_features(feature_config.get("spectral_features"))
    grouping_strategies = resolve_grouping_strategies(grouping)
    schemas = build_group_schemas(
        prepared.model_feature_names,
        strategies=grouping_strategies,
        feature_family_order=family_order,
    )

    background_mean, background_count = transformed_background_mean(
        normalizer=normalizer,
        background_batches=raw_feature_batches(
            prepared,
            prepared.background_samples,
            batch_size=batch_size,
        ),
        expected_width=expected_width,
    )
    explanation = explain_linear_log_odds_multi(
        pipeline=artifact.model,
        background_mean=background_mean,
        explanation_batches=raw_feature_batches(
            prepared,
            prepared.explanation_samples,
            batch_size=batch_size,
        ),
        group_schemas=schemas,
    )

    ctx = create_run_context(
        pipeline="explanation",
        output_root=Path(output_root).expanduser().resolve(),
        run_id=run_id,
        run_dir=output_run_dir,
        create_directories=("reports",),
    )
    config = {
        "schema_version": 2,
        "method": "linear_interventional_shap",
        "output_space": "log_odds_backdoored_vs_clean",
        "groupings": {name: list(schema.axes) for name, schema in schemas.items()},
        "shap": {
            "explainer": "LinearExplainer",
            "masker": "Independent",
            "background_representation": "coordinate_mean_in_fitted_model_space",
            "group_aggregation": "sum",
            "global_importance": "mean_absolute_grouped_shap",
        },
        "checkpoint": str(checkpoint),
        "feature_file": str(prepared.feature_path),
        "background_selection": str(background_selection),
        "explanation_selection": str(explanation_selection),
        "background_label": str(background_label),
        "background_max_samples": background_max_samples,
        "explanation_max_samples": explanation_max_samples,
        "random_state": int(random_state),
        "batch_size": int(batch_size),
        "plots": {
            "enabled": True,
            "max_display": int(plot_max_display),
            "max_local_plots": int(max_local_plots),
            "output_format": "png",
        },
        "background_sample_count": int(background_count),
        "explanation_sample_count": len(prepared.explanation_samples),
        "feature_configuration": feature_config,
        "source_metadata": str(prepared.source_metadata_path),
        "dataset_reference": str(prepared.dataset_reference_path),
    }
    report = {
        "run_id": str(ctx.run_id),
        "status": "complete",
        "method": "linear_interventional_shap",
        "output_space": "log_odds_backdoored_vs_clean",
        "background_sample_count": int(background_count),
        "explanation_sample_count": len(prepared.explanation_samples),
        "groupings": {
            name: {
                "axes": list(schema.axes),
                "n_groups": schema.n_groups,
            }
            for name, schema in schemas.items()
        },
        "max_additivity_error": float(explanation.max_additivity_error),
    }
    output_paths = write_multi_grouped_explanation_bundle(
        run_dir=ctx.run_dir,
        grouped_values=explanation.grouped_values,
        base_log_odds=explanation.base_log_odds,
        predicted_log_odds=explanation.predicted_log_odds,
        background_mean=explanation.background_mean,
        background_samples=prepared.background_samples,
        samples=prepared.explanation_samples,
        schemas=schemas,
        explanation_config=config,
        report=report,
        plot_max_display=plot_max_display,
        max_local_plots=max_local_plots,
    )
    for key, path in output_paths.items():
        ctx.add_artifact(key, path)
    ctx.finalize(
        {
            "pipeline": "explanation",
            "status": "complete",
            "method": "linear_interventional_shap",
            "explanation_config": str(output_paths["explanation_config"]),
        }
    )
    append_stage_timing(
        run_dir=ctx.run_dir,
        stage="linear_grouped_explanation",
        started_at=started_at,
        started_perf=started_perf,
    )
    experiment = experiment_context_from_stage_dir(ctx.run_dir)
    if experiment is not None:
        grouped_artifacts = {
            f"grouped_shap_values_{name.replace('+', '_')}": {
                "kind": "grouped_shap_values",
                "path": experiment.display_path(output_paths[f"{name.replace('+', '_')}_grouped_shap_values"]),
            }
            for name in schemas
        }
        experiment.update(
            stage="explanation",
            stage_status="completed",
            artifacts={
                **grouped_artifacts,
                "explanation_report": {
                    "kind": "explanation_report",
                    "path": experiment.display_path(output_paths["report"]),
                },
            },
        )
    return {
        "run_dir": str(ctx.run_dir),
        "report": str(output_paths["report"]),
        "grouped_shap_values": {
            name: str(output_paths[f"{name.replace('+', '_')}_grouped_shap_values"]) for name in schemas
        },
        "groupings": {
            name: {
                "global_importance_files": {
                    key.removeprefix(f"{name.replace('+', '_')}_global_importance_"): str(path)
                    for key, path in output_paths.items()
                    if key.startswith(f"{name.replace('+', '_')}_global_importance_")
                },
                "plot_files": {
                    key.removeprefix(f"{name.replace('+', '_')}_plot_"): str(path)
                    for key, path in output_paths.items()
                    if key.startswith(f"{name.replace('+', '_')}_plot_")
                },
            }
            for name in schemas
        },
    }

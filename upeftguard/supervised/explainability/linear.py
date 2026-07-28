"""Interventional grouped SHAP values for binary logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .grouping import GroupSchema


@dataclass(frozen=True)
class LinearGroupedExplanation:
    grouped_values: np.ndarray
    base_log_odds: np.ndarray
    predicted_log_odds: np.ndarray
    max_additivity_error: float
    background_mean: np.ndarray


@dataclass(frozen=True)
class LinearMultiGroupedExplanation:
    grouped_values: dict[str, np.ndarray]
    base_log_odds: np.ndarray
    predicted_log_odds: np.ndarray
    max_additivity_error: float
    background_mean: np.ndarray


def linear_pipeline_parts(pipeline: Any) -> tuple[Any, Any]:
    named_steps = getattr(pipeline, "named_steps", None)
    if not isinstance(named_steps, dict) and not hasattr(named_steps, "get"):
        raise ValueError("Linear SHAP requires a fitted sklearn Pipeline")
    normalizer = named_steps.get("normalizer")
    estimator = named_steps.get("model")
    if normalizer is None or estimator is None:
        raise ValueError("Expected checkpoint pipeline steps named 'normalizer' and 'model'")
    if estimator.__class__.__name__ != "LogisticRegression":
        raise ValueError(
            f"This explanation method currently supports LogisticRegression only; got {estimator.__class__.__name__}"
        )
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.shape != (2,) or not np.array_equal(classes, np.asarray([0, 1])):
        raise ValueError(f"Expected binary LogisticRegression classes [0, 1], got {classes.tolist()}")
    coefficients = np.asarray(getattr(estimator, "coef_", []))
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ValueError(f"Expected one binary logistic coefficient row, got shape={coefficients.shape}")
    return normalizer, estimator


def _transform_model_inputs(normalizer: Any, raw: np.ndarray) -> np.ndarray:
    if normalizer == "passthrough":
        return np.asarray(raw, dtype=np.float64)
    if not hasattr(normalizer, "transform"):
        raise ValueError(
            f"Expected the fitted pipeline normalizer to expose transform() or be 'passthrough'; got {normalizer!r}"
        )
    return np.asarray(normalizer.transform(raw), dtype=np.float64)


def transformed_background_mean(
    *,
    normalizer: Any,
    background_batches: Iterable[np.ndarray],
    expected_width: int,
) -> tuple[np.ndarray, int]:
    """Compute the selected background cohort's mean in fitted-model coordinates."""

    total = np.zeros(int(expected_width), dtype=np.float64)
    count = 0
    for raw_batch in background_batches:
        raw = np.asarray(raw_batch)
        if raw.ndim != 2 or int(raw.shape[1]) != int(expected_width):
            raise ValueError(f"Background batch shape {raw.shape} does not match model width {expected_width}")
        transformed = _transform_model_inputs(normalizer, raw)
        total += transformed.sum(axis=0, dtype=np.float64)
        count += int(transformed.shape[0])
    if count == 0:
        raise ValueError("Background selection contains zero eligible samples")
    return total / float(count), count


def aggregate_column_shap_values(values: np.ndarray, schema: GroupSchema) -> np.ndarray:
    """Sum raw-column SHAP values into the schema's disjoint groups."""

    raw_values = np.asarray(values, dtype=np.float64)
    if raw_values.ndim != 2 or int(raw_values.shape[1]) != schema.n_columns:
        raise ValueError(f"SHAP value shape {raw_values.shape} does not match grouped schema width {schema.n_columns}")
    order = np.argsort(schema.column_group_indices, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(schema.group_column_counts[:-1], dtype=np.int64)))
    return np.add.reduceat(raw_values[:, order], offsets, axis=1)


def explain_linear_log_odds(
    *,
    pipeline: Any,
    background_mean: np.ndarray,
    explanation_batches: Iterable[np.ndarray],
    group_schema: GroupSchema,
    additivity_tolerance: float = 1e-7,
) -> LinearGroupedExplanation:
    """Run ``shap.LinearExplainer`` for one coalition schema."""

    result = explain_linear_log_odds_multi(
        pipeline=pipeline,
        background_mean=background_mean,
        explanation_batches=explanation_batches,
        group_schemas={group_schema.name: group_schema},
        additivity_tolerance=additivity_tolerance,
    )
    return LinearGroupedExplanation(
        grouped_values=result.grouped_values[group_schema.name],
        base_log_odds=result.base_log_odds,
        predicted_log_odds=result.predicted_log_odds,
        max_additivity_error=result.max_additivity_error,
        background_mean=result.background_mean,
    )


def explain_linear_log_odds_multi(
    *,
    pipeline: Any,
    background_mean: np.ndarray,
    explanation_batches: Iterable[np.ndarray],
    group_schemas: dict[str, GroupSchema],
    additivity_tolerance: float = 1e-7,
) -> LinearMultiGroupedExplanation:
    """Compute raw linear SHAP once per batch and aggregate every requested schema."""

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on installation extra
        raise RuntimeError(
            "Grouped explanations require SHAP. Install the project environment with the explanation dependency."
        ) from exc

    normalizer, estimator = linear_pipeline_parts(pipeline)
    mean = np.asarray(background_mean, dtype=np.float64).reshape(-1)
    if not group_schemas:
        raise ValueError("At least one coalition schema is required")
    widths = {schema.n_columns for schema in group_schemas.values()}
    if len(widths) != 1 or mean.size not in widths:
        raise ValueError(f"Background mean width {mean.size} does not match coalition schema widths {sorted(widths)}")
    for name, schema in group_schemas.items():
        if name != schema.name:
            raise ValueError(f"Coalition schema key {name!r} does not match schema name {schema.name!r}")

    masker = shap.maskers.Independent(mean[None, :])
    explainer = shap.LinearExplainer(estimator, masker=masker, link=shap.links.identity)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    grouped_batches: dict[str, list[np.ndarray]] = {name: [] for name in group_schemas}
    base_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    max_error = 0.0
    for raw_batch in explanation_batches:
        raw = np.asarray(raw_batch)
        transformed = _transform_model_inputs(normalizer, raw)
        explanation = explainer(transformed)
        raw_values = np.asarray(explanation.values, dtype=np.float64)
        if raw_values.ndim != 2:
            raise ValueError(f"Expected 2D SHAP values for binary logistic regression, got {raw_values.shape}")
        grouped = {name: aggregate_column_shap_values(raw_values, schema) for name, schema in group_schemas.items()}
        logits = np.asarray(estimator.decision_function(transformed), dtype=np.float64).reshape(-1)
        first_values = next(iter(grouped.values()))
        reconstructed = expected_value + first_values.sum(axis=1, dtype=np.float64)
        error = float(np.max(np.abs(reconstructed - logits), initial=0.0))
        max_error = max(max_error, error)
        for name, values in grouped.items():
            grouped_batches[name].append(values)
        base_batches.append(np.full(logits.shape, expected_value, dtype=np.float64))
        logit_batches.append(logits)

    if not base_batches:
        raise ValueError("Explanation selection contains zero eligible samples")
    scale = max(1.0, float(np.max(np.abs(np.concatenate(logit_batches)))))
    if max_error > float(additivity_tolerance) * scale:
        raise ValueError(
            "Linear SHAP additivity check failed: "
            f"max_error={max_error:.6g}, tolerance={float(additivity_tolerance) * scale:.6g}"
        )

    return LinearMultiGroupedExplanation(
        grouped_values={name: np.concatenate(batches, axis=0) for name, batches in grouped_batches.items()},
        base_log_odds=np.concatenate(base_batches, axis=0),
        predicted_log_odds=np.concatenate(logit_batches, axis=0),
        max_additivity_error=max_error,
        background_mean=mean,
    )

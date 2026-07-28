"""Map emitted spectral columns to selectable coalition schemas."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import re
from typing import Any, Iterable

import numpy as np

from ...artifacts.schema import feature_group_for_feature_name
from ...contracts.spectral import feature_block_name, layer_identifier_for_block_name


GROUPING_AXES = ("layer", "block", "head", "feature")
DEFAULT_GROUPING = ("layer", "feature")

_HEAD_SUFFIX_RE = re.compile(r"\.head(\d+)$")
_BLOCK_ROLE_RE = re.compile(r"\.(q_proj|v_proj|qv_sum)\.")
_NATURAL_PART_RE = re.compile(r"(\d+)")
_BLOCK_LABELS = {
    "q_proj": "q",
    "v_proj": "v",
    "qv_sum": "q+v",
}


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in _NATURAL_PART_RE.split(str(value)))


def layer_for_feature_name(feature_name: str) -> str:
    """Return the layer identifier while discarding role and attention head."""

    block_name = feature_block_name(feature_name)
    module_block_name = _HEAD_SUFFIX_RE.sub("", block_name)
    layer_name = layer_identifier_for_block_name(module_block_name)
    if not layer_name:
        raise ValueError(f"Could not resolve a layer for spectral feature {feature_name!r}")
    return str(layer_name)


def block_for_feature_name(feature_name: str) -> str:
    """Return ``q``, ``v``, or ``q+v`` for one emitted spectral column."""

    match = _BLOCK_ROLE_RE.search(f".{feature_block_name(feature_name)}.")
    if match is None:
        raise ValueError(f"Could not resolve a block role for spectral feature {feature_name!r}")
    return _BLOCK_LABELS[match.group(1)]


def head_for_feature_name(feature_name: str) -> str:
    """Return the normalized attention-head label for one spectral column."""

    match = _HEAD_SUFFIX_RE.search(feature_block_name(feature_name))
    if match is None:
        raise ValueError(f"Could not resolve an attention head for spectral feature {feature_name!r}")
    width = max(2, len(match.group(1)))
    return f"head{int(match.group(1)):0{width}d}"


def spectral_feature_for_feature_name(feature_name: str) -> str:
    """Return the feature label while keeping each ``sv_k`` column separate."""

    family = feature_group_for_feature_name(feature_name)
    if family is None:
        raise ValueError(f"Unsupported spectral feature suffix in model column {feature_name!r}")
    if family != "sv_topk":
        return str(family)
    suffix = str(feature_name).rsplit(".", 1)[-1]
    if not re.fullmatch(r"sv_\d+", suffix):
        raise ValueError(f"Could not resolve a singular-value index for spectral feature {feature_name!r}")
    return suffix


def _axis_assignment(feature_name: str) -> dict[str, str]:
    return {
        "layer": layer_for_feature_name(feature_name),
        "block": block_for_feature_name(feature_name),
        "head": head_for_feature_name(feature_name),
        "feature": spectral_feature_for_feature_name(feature_name),
    }


def _canonical_axes(axes: Iterable[str]) -> tuple[str, ...]:
    cleaned = tuple(str(axis).strip().lower() for axis in axes if str(axis).strip())
    unknown = sorted(set(cleaned) - set(GROUPING_AXES))
    if unknown:
        raise ValueError(f"Unknown grouping axes {unknown}; expected a subset of {list(GROUPING_AXES)}")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"Grouping axes may not be repeated: {cleaned}")
    if not cleaned:
        raise ValueError("A grouping strategy must contain at least one axis")
    selected = set(cleaned)
    return tuple(axis for axis in GROUPING_AXES if axis in selected)


def grouping_name(axes: Iterable[str]) -> str:
    return "+".join(_canonical_axes(axes))


def resolve_grouping_strategies(
    requested: Iterable[str] | None,
) -> tuple[tuple[str, ...], ...]:
    """Resolve CLI grouping tokens, including ``all``, into canonical axis tuples."""

    tokens = [str(value).strip().lower() for value in (requested or []) if str(value).strip()]
    if not tokens:
        return (DEFAULT_GROUPING,)
    if "all" in tokens:
        if len(tokens) != 1:
            raise ValueError("--grouping all cannot be combined with explicit grouping strategies")
        return tuple(
            tuple(combo) for size in range(1, len(GROUPING_AXES) + 1) for combo in combinations(GROUPING_AXES, size)
        )

    strategies: list[tuple[str, ...]] = []
    for token in tokens:
        axes = _canonical_axes(token.split("+"))
        if axes not in strategies:
            strategies.append(axes)
    return tuple(strategies)


@dataclass(frozen=True)
class GroupSchema:
    """Immutable assignment of model columns to one coalition strategy."""

    axes: tuple[str, ...]
    axis_values: tuple[tuple[str, ...], ...]
    group_names: tuple[str, ...]
    group_keys: tuple[tuple[str, ...], ...]
    column_group_indices: np.ndarray
    group_column_counts: np.ndarray

    @property
    def name(self) -> str:
        return grouping_name(self.axes)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def n_columns(self) -> int:
        return int(self.column_group_indices.size)

    @property
    def n_dimensions(self) -> int:
        return len(self.axes)

    def values_for_axis(self, axis: str) -> tuple[str, ...]:
        try:
            return self.axis_values[self.axes.index(str(axis))]
        except ValueError:
            return ()

    def group_value(self, group_index: int, axis: str) -> str:
        try:
            position = self.axes.index(str(axis))
        except ValueError:
            return ""
        return self.group_keys[int(group_index)][position]

    @property
    def layer_names(self) -> tuple[str, ...]:
        return self.values_for_axis("layer")

    @property
    def feature_families(self) -> tuple[str, ...]:
        return self.values_for_axis("feature")

    @property
    def group_layers(self) -> tuple[str, ...]:
        return tuple(self.group_value(index, "layer") for index in range(self.n_groups))

    @property
    def group_feature_families(self) -> tuple[str, ...]:
        return tuple(self.group_value(index, "feature") for index in range(self.n_groups))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "grouping": list(self.axes),
            "grouping_name": self.name,
            "n_groups": self.n_groups,
            "n_model_columns": self.n_columns,
            "axis_values": {axis: list(values) for axis, values in zip(self.axes, self.axis_values)},
            "groups": [
                {
                    "group_index": index,
                    "group_name": self.group_names[index],
                    "dimensions": {axis: self.group_keys[index][position] for position, axis in enumerate(self.axes)},
                    "column_count": int(self.group_column_counts[index]),
                }
                for index in range(self.n_groups)
            ],
        }


def _ordered_feature_values(
    observed: set[str],
    *,
    feature_family_order: Iterable[str] | None,
) -> tuple[str, ...]:
    if feature_family_order is None:
        return tuple(sorted(observed, key=_natural_key))

    ordered: list[str] = []
    for family in (str(value) for value in feature_family_order):
        if family == "sv_topk":
            ordered.extend(sorted((value for value in observed if re.fullmatch(r"sv_\d+", value)), key=_natural_key))
        elif family in observed:
            ordered.append(family)
    ordered.extend(sorted(observed - set(ordered), key=_natural_key))
    return tuple(ordered)


def build_group_schema(
    feature_names: list[str] | tuple[str, ...],
    *,
    axes: Iterable[str],
    feature_family_order: Iterable[str] | None = None,
) -> GroupSchema:
    """Build and validate one complete Cartesian coalition schema."""

    names = [str(name) for name in feature_names]
    if not names:
        raise ValueError("Cannot build a grouped explanation schema from zero feature columns")
    resolved_axes = _canonical_axes(axes)
    assignments = [_axis_assignment(name) for name in names]

    values_by_axis: dict[str, tuple[str, ...]] = {}
    for axis in resolved_axes:
        observed = {assignment[axis] for assignment in assignments}
        if axis == "feature":
            values_by_axis[axis] = _ordered_feature_values(
                observed,
                feature_family_order=feature_family_order,
            )
        elif axis == "block":
            values_by_axis[axis] = tuple(label for label in ("q", "v", "q+v") if label in observed)
        else:
            values_by_axis[axis] = tuple(sorted(observed, key=_natural_key))

    axis_values = tuple(values_by_axis[axis] for axis in resolved_axes)
    group_keys = tuple(product(*axis_values))
    group_index = {key: index for index, key in enumerate(group_keys)}
    column_keys = [tuple(assignment[axis] for axis in resolved_axes) for assignment in assignments]
    column_groups = np.asarray([group_index[key] for key in column_keys], dtype=np.int64)
    counts = np.bincount(column_groups, minlength=len(group_keys)).astype(np.int64, copy=False)
    missing = [group_keys[index] for index, count in enumerate(counts) if int(count) == 0]
    if missing:
        preview = ", ".join("::".join(key) for key in missing[:5])
        raise ValueError(f"Raw feature schema is not a complete coalition grid. Missing groups: {preview}")

    return GroupSchema(
        axes=resolved_axes,
        axis_values=axis_values,
        group_names=tuple("::".join(key) for key in group_keys),
        group_keys=group_keys,
        column_group_indices=column_groups,
        group_column_counts=counts,
    )


def build_group_schemas(
    feature_names: list[str] | tuple[str, ...],
    *,
    strategies: Iterable[Iterable[str]],
    feature_family_order: Iterable[str] | None = None,
) -> dict[str, GroupSchema]:
    """Build every requested coalition schema from the same model-column layout."""

    schemas: dict[str, GroupSchema] = {}
    for axes in strategies:
        schema = build_group_schema(
            feature_names,
            axes=axes,
            feature_family_order=feature_family_order,
        )
        schemas[schema.name] = schema
    if not schemas:
        raise ValueError("At least one grouping strategy is required")
    return schemas


def build_layer_feature_group_schema(
    feature_names: list[str] | tuple[str, ...],
    *,
    feature_family_order: list[str] | tuple[str, ...] | None = None,
) -> GroupSchema:
    """Compatibility wrapper for the default ``layer+feature`` strategy."""

    return build_group_schema(
        feature_names,
        axes=DEFAULT_GROUPING,
        feature_family_order=feature_family_order,
    )

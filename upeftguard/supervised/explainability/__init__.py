"""Grouped explanations for supervised UPEFTGuard models."""

from .grouping import (
    GroupSchema,
    build_group_schema,
    build_group_schemas,
    build_layer_feature_group_schema,
    resolve_grouping_strategies,
)
from .linear import (
    LinearGroupedExplanation,
    LinearMultiGroupedExplanation,
    explain_linear_log_odds,
    explain_linear_log_odds_multi,
)

__all__ = [
    "GroupSchema",
    "LinearGroupedExplanation",
    "LinearMultiGroupedExplanation",
    "build_group_schema",
    "build_group_schemas",
    "build_layer_feature_group_schema",
    "explain_linear_log_odds",
    "explain_linear_log_odds_multi",
    "resolve_grouping_strategies",
]

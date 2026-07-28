from __future__ import annotations

import argparse

from .commands.artifacts import (
    handle_aggregate_features,
    handle_export_feature_subset,
    handle_merge_features,
    register_artifact_commands,
)
from .commands.experiment import (
    handle_aggregate,
    handle_extract,
    handle_explain,
    handle_full,
    handle_infer,
    handle_train,
    register_experiment_commands,
)
from .commands.internal import (
    handle_feature_extract,
    handle_supervised,
    register_internal_commands,
)
from .supervised.lifecycle.pipeline import run_supervised_pipeline
from .workflows import (
    run_checkpoint_inference,
    run_feature_aggregation,
    run_feature_extraction,
    run_grouped_explanation,
    run_full_experiment,
    run_supervised_training,
)


def _cmd_experiment_extract(args: argparse.Namespace) -> int:
    return handle_extract(args, run_feature_extraction)


def _cmd_experiment_aggregate(args: argparse.Namespace) -> int:
    return handle_aggregate(args, run_feature_aggregation)


def _cmd_experiment_train(args: argparse.Namespace) -> int:
    return handle_train(args, run_supervised_training)


def _cmd_experiment_infer(args: argparse.Namespace) -> int:
    return handle_infer(args, run_checkpoint_inference)


def _cmd_experiment_explain(args: argparse.Namespace) -> int:
    return handle_explain(args, run_grouped_explanation)


def _cmd_experiment_full(args: argparse.Namespace) -> int:
    return handle_full(args, run_full_experiment)


def _cmd_feature_extract(args: argparse.Namespace) -> int:
    return handle_feature_extract(args, run_feature_extraction)


def _cmd_run_supervised(args: argparse.Namespace) -> int:
    return handle_supervised(args, run_supervised_pipeline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="upeftguard experiment CLI")
    sub = parser.add_subparsers(dest="command_group", required=True)
    register_experiment_commands(
        sub,
        {
            "full": _cmd_experiment_full,
            "extract": _cmd_experiment_extract,
            "aggregate": _cmd_experiment_aggregate,
            "train": _cmd_experiment_train,
            "infer": _cmd_experiment_infer,
            "explain": _cmd_experiment_explain,
        },
    )
    register_internal_commands(
        sub,
        {
            "supervised": _cmd_run_supervised,
            "feature_extract": _cmd_feature_extract,
        },
    )
    register_artifact_commands(
        sub,
        {
            "merge_features": handle_merge_features,
            "export_subset": handle_export_feature_subset,
            "aggregate_features": handle_aggregate_features,
        },
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

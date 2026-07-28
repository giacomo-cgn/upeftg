"""Public experiment command handlers and parser registration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..features.spectral import (
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES,
)
from .common import (
    _add_backend_args,
    _add_common_run_args,
    _add_spectral_args,
    _add_training_args,
    _print_result,
    _resolve_output_root,
)


def handle_extract(args: argparse.Namespace, operation) -> int:
    result = operation(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        nodes=args.nodes,
        cpus_per_worker=args.cpus_per_worker,
        workers_per_node=args.workers_per_node,
        dry_run=args.dry_run,
        features=args.features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        parallelization_settings=args.parallelization_settings,
    )
    _print_result(result)
    return 0


def handle_aggregate(args: argparse.Namespace, operation) -> int:
    result = operation(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        cpus_per_worker=args.cpus_per_worker,
        dry_run=args.dry_run,
        feature_root=args.feature_root,
        features=args.features,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
    )
    _print_result(result)
    return 0


def handle_train(args: argparse.Namespace, operation) -> int:
    result = operation(
        manifest_json=args.manifest_json,
        feature_file=args.feature_file,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        nodes=args.nodes,
        cpus_per_worker=args.cpus_per_worker,
        workers_per_node=args.workers_per_node,
        dry_run=args.dry_run,
        features=args.features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        cv_folds=args.cv_folds,
        cv_strategy=args.cv_strategy,
        input_normalization=args.input_normalization,
        cv_derived_refit_epochs=args.cv_derived_refit_epochs,
        no_refit=args.no_refit,
        random_state=args.random_state,
        train_split=args.train_split,
        calibration_split=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_seeds=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=None,
        model_name=args.model,
        hyperparams=args.hyperparams,
        checkpoint_interval_hours=args.checkpoint_interval_hours,
        resume_checkpoint=args.resume_checkpoint,
        task_mode=args.task_mode,
        multiclass_attack_names=args.multiclass_attack_names,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        selection_metric=args.selection_metric,
    )
    _print_result(result)
    return 0


def handle_infer(args: argparse.Namespace, operation) -> int:
    result = operation(
        checkpoint=args.checkpoint,
        manifest_json=args.manifest_json,
        feature_file=args.feature_file,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        cpus_per_worker=args.cpus_per_worker,
        dry_run=args.dry_run,
    )
    _print_result(result)
    return 0


def handle_explain(args: argparse.Namespace, operation) -> int:
    result = operation(
        checkpoint=args.checkpoint,
        feature_file=args.feature_file,
        background_selection=args.background_selection,
        explanation_selection=args.explanation_selection,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        partition=args.partition,
        cpus_per_worker=args.cpus_per_worker,
        dry_run=args.dry_run,
        background_label=args.background_label,
        background_max_samples=args.background_max_samples,
        explanation_max_samples=args.explanation_max_samples,
        random_state=args.random_state,
        batch_size=args.batch_size,
        grouping=args.grouping,
        plot_max_display=args.plot_max_display,
        max_local_plots=args.max_local_plots,
    )
    _print_result(result)
    return 0


def handle_full(args: argparse.Namespace, operation) -> int:
    result = operation(
        manifest_json=args.manifest_json,
        train_manifest_json=args.train_manifest_json,
        inference_manifest_json=args.infer_manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        backend=args.backend,
        partition=args.partition,
        cpus_per_worker=args.cpus_per_worker,
        extract_nodes=args.extract_nodes,
        extract_cpus_per_worker=args.extract_cpus_per_worker,
        extract_workers_per_node=args.extract_workers_per_node,
        extract_parallelization_settings=args.extract_parallelization_settings,
        aggregate_cpus_per_worker=args.aggregate_cpus_per_worker,
        train_nodes=args.train_nodes,
        train_cpus_per_worker=args.train_cpus_per_worker,
        train_workers_per_node=args.train_workers_per_node,
        dry_run=args.dry_run,
        features=args.features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        cv_folds=args.cv_folds,
        cv_strategy=args.cv_strategy,
        input_normalization=args.input_normalization,
        cv_derived_refit_epochs=args.cv_derived_refit_epochs,
        no_refit=args.no_refit,
        random_state=args.random_state,
        train_split=args.train_split,
        calibration_split=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_seeds=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=None,
        model_name=args.model,
        hyperparams=args.hyperparams,
        checkpoint_interval_hours=args.checkpoint_interval_hours,
        resume_checkpoint=args.resume_checkpoint,
        task_mode=args.task_mode,
        multiclass_attack_names=args.multiclass_attack_names,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        selection_metric=args.selection_metric,
    )
    _print_result(result)
    return 0


def register_experiment_commands(sub, handlers: dict[str, object]) -> None:
    _register_pipeline_group(
        sub,
        group_name="experiment",
        group_help="Run model-neutral extraction, training, and inference workflows",
        command_dest="experiment_command",
        handlers=handlers,
    )


def _register_pipeline_group(
    sub,
    *,
    group_name: str,
    group_help: str,
    command_dest: str,
    handlers: dict[str, object],
) -> None:
    pipeline = sub.add_parser(group_name, help=group_help)
    pipeline_sub = pipeline.add_subparsers(dest=command_dest, required=True)

    full = pipeline_sub.add_parser(
        "full",
        help="Run extraction, optional aggregation, training, and inference sequentially",
    )
    _add_backend_args(full)
    full.add_argument("--extract-nodes", default="auto")
    full.add_argument("--extract-cpus-per-worker", default=None)
    full.add_argument("--extract-workers-per-node", default="auto")
    full.add_argument(
        "--extract-parallelization-settings",
        nargs="+",
        default=None,
        metavar="RANK:NODES:CPUS_PER_WORKER:WORKERS_PER_NODE",
        help="Rank-specific extraction settings; unlisted ranks use the discovered default allocation.",
    )
    full.add_argument("--aggregate-cpus-per-worker", default=None)
    full.add_argument("--train-nodes", default="auto")
    full.add_argument("--train-cpus-per-worker", default=None)
    full.add_argument("--train-workers-per-node", default="auto")
    _add_common_run_args(full, manifest_required=False)
    full.add_argument("--train-manifest-json", type=Path, default=None)
    full.add_argument("--infer-manifest-json", type=Path, default=None)
    _add_spectral_args(full, default_qv_mode=DEFAULT_SPECTRAL_QV_SUM_MODE)
    _add_training_args(full)
    full.set_defaults(func=handlers["full"])

    extract = pipeline_sub.add_parser("extract", help="Run spectral feature extraction")
    _add_backend_args(extract)
    _add_common_run_args(extract)
    _add_spectral_args(extract, default_qv_mode=DEFAULT_SPECTRAL_QV_SUM_MODE)
    extract.add_argument("--nodes", default="auto")
    extract.add_argument("--workers-per-node", default="auto")
    extract.add_argument(
        "--parallelization-settings",
        nargs="+",
        default=None,
        metavar="RANK:NODES:CPUS_PER_WORKER:WORKERS_PER_NODE",
        help="Rank overrides; every omitted rank uses the default node/worker allocation.",
    )
    extract.set_defaults(func=handlers["extract"])

    aggregate = pipeline_sub.add_parser(
        "aggregate",
        help="Build a model-neutral layer-sequence feature bundle",
    )
    _add_backend_args(aggregate)
    aggregate.add_argument("--feature-file", type=Path, required=True)
    aggregate.add_argument("--output-filename", type=Path, default=None)
    aggregate.add_argument("--output-root", type=Path, default=Path("runs"))
    aggregate.add_argument("--run-id", type=str, default=None)
    aggregate.add_argument("--feature-root", type=Path, default=None)
    aggregate.add_argument("--features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    aggregate.add_argument("--spectral-qv-sum-mode", choices=["none", "append", "only"], default="append")
    aggregate.add_argument(
        "--spectral-attention-granularity",
        choices=list(SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES),
        default=None,
    )
    aggregate.set_defaults(func=handlers["aggregate"])

    train = pipeline_sub.add_parser(
        "train",
        help="Train and finalize a selected supervised backend",
    )
    _add_backend_args(train)
    _add_common_run_args(train)
    train.add_argument("--feature-file", type=Path, required=True)
    _add_spectral_args(train, default_qv_mode="append")
    _add_training_args(train)
    train.add_argument("--nodes", default="auto")
    train.add_argument("--workers-per-node", default="auto")
    train.set_defaults(func=handlers["train"])

    infer = pipeline_sub.add_parser(
        "infer",
        help="Run inference from a checkpoint on an explicit inference manifest",
    )
    _add_backend_args(infer)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--manifest-json", type=Path, required=True)
    infer.add_argument("--feature-file", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, default=Path("runs"))
    infer.add_argument("--run-id", type=str, default=None)
    infer.set_defaults(func=handlers["infer"])

    explain = pipeline_sub.add_parser(
        "explain",
        help="Submit grouped linear SHAP explanations for a binary logistic checkpoint",
    )
    explain.add_argument("--checkpoint", type=Path, required=True)
    explain.add_argument("--feature-file", type=Path, required=True)
    explain.add_argument("--background-selection", type=Path, required=True)
    explain.add_argument("--explanation-selection", type=Path, required=True)
    explain.add_argument("--background-label", choices=["all", "clean", "backdoor"], default="all")
    explain.add_argument("--background-max-samples", type=int, default=None)
    explain.add_argument("--explanation-max-samples", type=int, default=None)
    explain.add_argument("--random-state", type=int, default=42)
    explain.add_argument("--batch-size", type=int, default=128)
    explain.add_argument(
        "--grouping",
        nargs="+",
        default=["layer+feature"],
        help=("Coalition strategies using layer, block, head, and feature axes; combine axes with '+' or pass 'all'"),
    )
    explain.add_argument("--plot-max-display", type=int, default=20)
    explain.add_argument("--max-local-plots", type=int, default=20)
    explain.add_argument("--output-root", type=Path, default=Path("runs"))
    explain.add_argument("--run-id", type=str, default=None)
    explain.add_argument("--partition", default="extra")
    explain.add_argument("--cpus-per-worker", default="4")
    explain.add_argument("--dry-run", action="store_true")
    explain.set_defaults(func=handlers["explain"])

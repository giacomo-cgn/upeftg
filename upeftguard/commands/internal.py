"""Internal feature and supervised worker commands."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from ..features.spectral import (
    DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES,
    SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES,
)
from ..utilities.core.paths import default_dataset_root, dataset_root_help
from ..orchestration.slurm import resolve_slurm_task_index
from .common import _add_supervised_internal_args, _print_result, _resolve_output_root


def handle_slurm_extraction_controller(args: argparse.Namespace) -> int:
    from ..orchestration.slurm.extraction import run_extraction_slurm_controller

    _print_result(run_extraction_slurm_controller(args.config))
    return 0


def handle_slurm_supervised_controller(args: argparse.Namespace) -> int:
    from ..orchestration.slurm.supervised import run_supervised_slurm_controller

    _print_result(run_supervised_slurm_controller(args.config))
    return 0


def handle_schema_shard_worker(args: argparse.Namespace) -> int:
    from ..orchestration.slurm.shard_worker import run_schema_shard_worker

    run_schema_shard_worker(
        schema_report_path=args.schema_report_path,
        shard_index=args.shard_index,
        dataset_root=args.dataset_root,
        features=list(args.features),
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        group_ids=args.group_ids,
    )
    return 0


def handle_merge_schema_group_shards(args: argparse.Namespace) -> int:
    from ..orchestration.slurm.merge_jobs import merge_schema_group_shards

    merge_schema_group_shards(
        schema_report_path=args.schema_report_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        pipeline_start_epoch_seconds=args.pipeline_start_epoch_seconds,
    )
    return 0


def handle_merge_spectral_shards(args: argparse.Namespace) -> int:
    from ..artifacts.merge.shards import merge_spectral_shards

    shard_run_dirs = list(args.shard_run_dirs or [])
    if args.shard_run_dir_glob is not None:
        shard_run_dirs = [Path(value).expanduser().resolve() for value in sorted(glob.glob(args.shard_run_dir_glob))]
        if not shard_run_dirs:
            raise FileNotFoundError(f"No shard run directories matched glob: {args.shard_run_dir_glob}")
    _print_result(
        merge_spectral_shards(
            manifest_json=args.manifest_json,
            dataset_root=args.dataset_root,
            shard_run_dirs=shard_run_dirs,
            output_dir=args.output_dir,
            pipeline_start_epoch_seconds=args.pipeline_start_epoch_seconds,
        )
    )
    return 0


def handle_finalize_schema_group_merge(args: argparse.Namespace) -> int:
    from ..artifacts.merge.files import finalize_schema_group_merge

    _print_result(
        finalize_schema_group_merge(
            schema_report_path=args.schema_report_path,
            output_dir=args.output_dir,
        )
    )
    return 0


def handle_prepare_spectral_shards(args: argparse.Namespace) -> int:
    from ..orchestration.sharding.planning import prepare_schema_sharded_manifests

    result = prepare_schema_sharded_manifests(
        manifest_path=args.manifest_json,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        worker_capacity=args.worker_capacity,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        report_path=args.report_path,
    )
    print(json.dumps(result, indent=2))
    return 0


def handle_feature_extract(args: argparse.Namespace, operation) -> int:
    if args.extractor != "spectral":
        raise ValueError("Only the stable spectral extractor is available in the active CLI")
    result = operation(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        backend="local",
        features=args.spectral_features,
        spectral_sv_top_k=args.spectral_sv_top_k,
        spectral_moment_source=args.spectral_moment_source,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        spectral_entrywise_delta_mode=args.spectral_entrywise_delta_mode,
        spectral_attention_granularity=args.spectral_attention_granularity,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
    )
    _print_result(result)
    return 0


def handle_supervised(args: argparse.Namespace, operation) -> int:
    task_index = resolve_slurm_task_index(args.task_index) if args.stage == "worker" else args.task_index
    if task_index is not None:
        task_index += int(args.task_index_offset)
    output_root = _resolve_output_root(args.output_root)
    prepared_run_dir = args.prepared_run_dir
    if prepared_run_dir is None and args.stage in {"all", "prepare"} and args.run_id is not None:
        prepared_run_dir = output_root / args.run_id / "training"
    if args.checkpoint_interval_hours is not None and prepared_run_dir is None:
        raise ValueError("--run-id is required with --checkpoint-interval-hours")
    result = operation(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=output_root,
        run_id=args.run_id,
        model_name=args.model,
        spectral_features=(list(args.features) if args.features is not None else None),
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
        train_split_percent=args.train_split,
        calibration_split_percent=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_random_states=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=None,
        tuning_executor=args.tuning_executor,
        slurm_partition=args.slurm_partition,
        cpus_per_worker=args.cpus_per_worker,
        stage=args.stage,
        run_dir=args.run_dir,
        task_index=task_index,
        task_mode=args.task_mode,
        multiclass_attack_names=(
            list(args.multiclass_attack_names) if args.multiclass_attack_names is not None else None
        ),
        hyperparams=args.hyperparams,
        transformer_checkpoint_dir=(
            Path(args.run_dir or prepared_run_dir) / "models" / "interval_checkpoints"
            if args.checkpoint_interval_hours is not None and (args.run_dir is not None or prepared_run_dir is not None)
            else None
        ),
        transformer_checkpoint_interval_seconds=(
            float(args.checkpoint_interval_hours) * 3600.0 if args.checkpoint_interval_hours is not None else None
        ),
        transformer_resume_checkpoint=args.resume_checkpoint,
        prepared_run_dir=prepared_run_dir,
        dann_source_rank=args.dann_source_rank,
        dann_target_adaptation_percent=args.dann_target_adaptation_percent,
        dann_lambda_max=args.dann_lambda_max,
        dann_lambda_gamma=args.dann_lambda_gamma,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        selection_metric=args.selection_metric,
        feature_file=args.feature_file,
    )
    _print_result(result)
    if result.get("next_steps"):
        print("Next steps:")
        for step in result["next_steps"]:
            print(f"- {step}")
    return 0


def handle_explanation_worker(args: argparse.Namespace) -> int:
    matplotlib_config_dir = Path(args.output_run_dir) / ".work" / "matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    from ..supervised.lifecycle.explanation import run_linear_grouped_explanation

    result = run_linear_grouped_explanation(
        checkpoint=args.checkpoint,
        feature_file=args.feature_file,
        background_selection=args.background_selection,
        explanation_selection=args.explanation_selection,
        output_root=_resolve_output_root(args.output_root),
        run_id=args.run_id,
        output_run_dir=args.output_run_dir,
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


def register_internal_commands(sub, handlers: dict[str, object]) -> None:
    run_parser = sub.add_parser("run", help="Internal compatibility runners")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)
    supervised = run_sub.add_parser("supervised", help="Internal supervised runner")
    _add_supervised_internal_args(supervised)
    supervised.set_defaults(func=handlers["supervised"])

    explanation = run_sub.add_parser(
        "explanation-worker",
        help="Internal compute-node worker for grouped linear SHAP",
    )
    explanation.add_argument("--checkpoint", type=Path, required=True)
    explanation.add_argument("--feature-file", type=Path, required=True)
    explanation.add_argument("--background-selection", type=Path, required=True)
    explanation.add_argument("--explanation-selection", type=Path, required=True)
    explanation.add_argument("--output-root", type=Path, default=Path("runs"))
    explanation.add_argument("--output-run-dir", type=Path, required=True)
    explanation.add_argument("--run-id", type=str, required=True)
    explanation.add_argument("--background-label", choices=["all", "clean", "backdoor"], default="all")
    explanation.add_argument("--background-max-samples", type=int, default=None)
    explanation.add_argument("--explanation-max-samples", type=int, default=None)
    explanation.add_argument("--random-state", type=int, default=42)
    explanation.add_argument("--batch-size", type=int, default=128)
    explanation.add_argument("--grouping", nargs="+", default=["layer+feature"])
    explanation.add_argument("--plot-max-display", type=int, default=20)
    explanation.add_argument("--max-local-plots", type=int, default=20)
    explanation.set_defaults(func=handle_explanation_worker)

    extraction_controller = run_sub.add_parser(
        "slurm-extraction-controller",
        help="Submit the feature-extraction Slurm worker graph",
    )
    extraction_controller.add_argument("config", type=Path)
    extraction_controller.set_defaults(func=handle_slurm_extraction_controller)

    supervised_controller = run_sub.add_parser(
        "slurm-supervised-controller",
        help="Submit the supervised Slurm worker graph",
    )
    supervised_controller.add_argument("config", type=Path)
    supervised_controller.set_defaults(func=handle_slurm_supervised_controller)

    shard_worker = run_sub.add_parser(
        "schema-shard-worker",
        help="Extract one shard index across compatible schema groups",
    )
    shard_worker.add_argument("--schema-report-path", type=Path, required=True)
    shard_worker.add_argument("--shard-index", type=int)
    shard_worker.add_argument("--dataset-root", type=Path, required=True)
    shard_worker.add_argument("--features", nargs="+", required=True)
    shard_worker.add_argument("--spectral-sv-top-k", type=int, required=True)
    shard_worker.add_argument("--spectral-moment-source", required=True)
    shard_worker.add_argument("--spectral-entrywise-delta-mode", required=True)
    shard_worker.add_argument("--spectral-attention-granularity", required=True)
    shard_worker.add_argument("--stream-block-size", type=int, required=True)
    shard_worker.add_argument("--dtype", required=True)
    shard_worker.add_argument("--group-ids", nargs="+", default=None)
    shard_worker.set_defaults(func=handle_schema_shard_worker)

    merge_groups = run_sub.add_parser(
        "merge-schema-group-shards",
        help="Merge shard outputs for every schema group",
    )
    merge_groups.add_argument("--schema-report-path", type=Path, required=True)
    merge_groups.add_argument("--dataset-root", type=Path, required=True)
    merge_groups.add_argument("--output-dir", type=Path, required=True)
    merge_groups.add_argument("--pipeline-start-epoch-seconds", type=float, required=True)
    merge_groups.set_defaults(func=handle_merge_schema_group_shards)

    merge_shards = run_sub.add_parser(
        "merge-spectral-shards",
        help="Merge sharded spectral extraction artifacts",
    )
    merge_shards.add_argument("--manifest-json", type=Path, required=True)
    merge_shards.add_argument("--dataset-root", type=Path, required=True)
    merge_shards.add_argument("--output-dir", type=Path, required=True)
    merge_shards.add_argument("--pipeline-start-epoch-seconds", type=float)
    shard_source = merge_shards.add_mutually_exclusive_group(required=True)
    shard_source.add_argument("--shard-run-dirs", type=Path, nargs="+")
    shard_source.add_argument("--shard-run-dir-glob")
    merge_shards.set_defaults(func=handle_merge_spectral_shards)

    finalize_groups = run_sub.add_parser(
        "finalize-schema-group-merge",
        help="Merge all schema-group artifacts into one feature bundle",
    )
    finalize_groups.add_argument("--schema-report-path", type=Path, required=True)
    finalize_groups.add_argument("--output-dir", type=Path, required=True)
    finalize_groups.set_defaults(func=handle_finalize_schema_group_merge)

    prepare_shards = run_sub.add_parser(
        "prepare-spectral-shards",
        help="Prepare schema-aware shard manifests",
    )
    prepare_shards.add_argument("--manifest-json", type=Path, required=True)
    prepare_shards.add_argument("--dataset-root", type=Path, required=True)
    prepare_shards.add_argument("--output-dir", type=Path, required=True)
    prepare_shards.add_argument("--worker-capacity", type=int, required=True)
    prepare_shards.add_argument("--spectral-qv-sum-mode", default="none")
    prepare_shards.add_argument("--report-path", type=Path)
    prepare_shards.set_defaults(func=handle_prepare_spectral_shards)

    feature_parser = sub.add_parser("feature", help="Internal feature commands")
    feature_sub = feature_parser.add_subparsers(dest="feature_command", required=True)
    feature_extract = feature_sub.add_parser("extract", help="Internal spectral feature extraction")
    feature_extract.add_argument("--manifest-json", type=Path, required=True)
    feature_extract.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help=dataset_root_help())
    feature_extract.add_argument("--extractor", choices=["spectral"], default="spectral")
    feature_extract.add_argument("--output-root", type=Path, default=Path("runs"))
    feature_extract.add_argument("--run-id", type=str, default=None)
    feature_extract.add_argument("--spectral-features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    feature_extract.add_argument("--spectral-sv-top-k", type=int, default=8)
    feature_extract.add_argument(
        "--spectral-moment-source", choices=["entrywise", "sv", "both"], default=DEFAULT_SPECTRAL_MOMENT_SOURCE
    )
    feature_extract.add_argument(
        "--spectral-qv-sum-mode", choices=["none", "append", "only"], default=DEFAULT_SPECTRAL_QV_SUM_MODE
    )
    feature_extract.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    feature_extract.add_argument(
        "--spectral-attention-granularity",
        choices=list(SUPPORTED_SPECTRAL_ATTENTION_GRANULARITIES),
        default=DEFAULT_SPECTRAL_ATTENTION_GRANULARITY,
    )
    feature_extract.add_argument("--stream-block-size", type=int, default=131072)
    feature_extract.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    feature_extract.set_defaults(func=handlers["feature_extract"])

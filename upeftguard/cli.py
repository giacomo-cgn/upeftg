from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from .clustering.pipeline import run_clustering_pipeline
from .features.registry import extract_features, supported_extractors
from .features.spectral import (
    DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_SPECTRAL_MOMENT_SOURCE,
    DEFAULT_SPECTRAL_QV_SUM_MODE,
    SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES,
)
from .supervised.pipeline import (
    DANN_DEFAULT_LAMBDA_GAMMA,
    DANN_DEFAULT_LAMBDA_MAX,
    DANN_DEFAULT_LR_ALPHA,
    DANN_DEFAULT_LR_BETA,
    DANN_DEFAULT_SOURCE_RANK,
    DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
    SUPPORTED_SELECTION_METRICS,
    run_supervised_pipeline,
)
from .supervised.registry import (
    default_autoencoder_hyperparams_path,
    default_cnn_hyperparams_path,
    default_few_shot_hyperparams_path,
    registered_models,
)
from .unsupervised.analysis import (
    run_supervised_cnn_feature_tsne_pipeline,
    run_unsupervised_layer_scatter_pipeline,
    run_unsupervised_rank_feature_values_pipeline,
    run_unsupervised_tsne_pipeline,
)
from .unsupervised.gmm_train_inference import run_gmm_train_inference_pipeline
from .utilities.artifacts.dataset_references import (
    build_dataset_reference_payload_from_items,
    default_dataset_reference_report_path,
    write_dataset_reference_report,
)
from .utilities.artifacts.aggregate_features import aggregate_features
from .utilities.artifacts.export_feature_subset import export_feature_subset
from .utilities.core.manifest import parse_single_manifest_json, resolve_manifest_path
from .utilities.core.paths import default_dataset_root, dataset_root_help
from .utilities.core.run_context import create_run_context
from .utilities.core.serialization import json_ready
from .utilities.merge.merge_feature_files import merge_feature_files


EXPERIMENT_COMMANDS: dict[str, tuple[str, str]] = {
    "paper-qv-reference": (
        "upeftguard.experiments.paper_qv_reference",
        "Run the paper q/v reference experiment",
    ),
    "rank-normalization-study": (
        "upeftguard.experiments.rank_normalization_study",
        "Run or report the rank-normalization study",
    ),
    "supervised-results-summary": (
        "upeftguard.experiments.supervised_architecture_breakdown",
        "Generate supervised results_summary.md outputs",
    ),
    "imdb-clean-reference": (
        "upeftguard.experiments.imdb_clean_reference",
        "Run the IMDb clean-reference anomaly experiment",
    ),
    "backdoor-detection-summaries": (
        "upeftguard.experiments.backdoor_detection_summaries",
        "Aggregate completed supervised run outputs",
    ),
    "supervised-slurm": (
        "upeftguard.experiments.supervised_slurm",
        "Submit the generic supervised Slurm workflow",
    ),
    "prepare-attack-family-leave-one-out": (
        "upeftguard.experiments.attack_family_leave_one_out",
        "Generate attack-family leave-one-out manifests",
    ),
    "prepare-group-leave-one-out": (
        "upeftguard.experiments.group_leave_one_out",
        "Generate adapter or architecture leave-one-out manifests",
    ),
    "supervised-cnn-suite": (
        "upeftguard.experiments.supervised_cnn_suite",
        "Run a supervised CNN transfer suite",
    ),
}


def _execution_root() -> Path:
    return Path.cwd().resolve()


def _is_filesystem_root(path: Path) -> bool:
    return path == path.parent


def _resolve_output_root(output_root: Path, caller: str) -> Path:
    root = _execution_root()
    candidate = output_root.expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved == root or _is_filesystem_root(resolved):
        # Never write caller artifacts directly at execution root or filesystem root.
        return (root / "runs" / caller).resolve()
    return resolved


def _has_option(tokens: list[str], option: str) -> bool:
    return any(tok == option or tok.startswith(option + "=") for tok in tokens)


def _parse_learning_rate(raw: str) -> str | float:
    value = str(raw).strip()
    if value.lower() == "auto":
        return "auto"
    return float(value)


def _rewrite_download_local_dir(tokens: list[str]) -> list[str]:
    root = _execution_root()
    default_download_root = default_dataset_root()
    safe_local_root = (root / "data").resolve()
    out = list(tokens)

    if not _has_option(out, "--local-dir"):
        return ["--local-dir", str(default_download_root), *out]

    for idx, tok in enumerate(out):
        if tok == "--local-dir" and idx + 1 < len(out):
            current = Path(out[idx + 1]).expanduser()
            resolved = (current if current.is_absolute() else root / current).resolve()
            if resolved == root or _is_filesystem_root(resolved):
                out[idx + 1] = str(safe_local_root)
            return out
        if tok.startswith("--local-dir="):
            raw = tok.split("=", 1)[1]
            current = Path(raw).expanduser()
            resolved = (current if current.is_absolute() else root / current).resolve()
            if resolved == root or _is_filesystem_root(resolved):
                out[idx] = f"--local-dir={safe_local_root}"
            return out
    return out


def _normalize_download_args(tokens: list[str]) -> list[str]:
    out = list(tokens)
    while out and out[0] == "--":
        out = out[1:]
    return out


def _extractor_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    extractor_name = str(getattr(args, "extractor", "svd"))
    spectral_features = getattr(args, "spectral_features", None)
    if spectral_features is not None:
        spectral_features = list(spectral_features)

    if extractor_name == "spectral":
        return {
            "block_size": int(getattr(args, "stream_block_size", 131072)),
            "dtype": str(getattr(args, "dtype", "float32")),
            "spectral_features": spectral_features,
            "spectral_sv_top_k": int(getattr(args, "spectral_sv_top_k", 8)),
            "spectral_moment_source": str(
                getattr(args, "spectral_moment_source", DEFAULT_SPECTRAL_MOMENT_SOURCE)
            ),
            "spectral_qv_sum_mode": str(getattr(args, "spectral_qv_sum_mode", DEFAULT_SPECTRAL_QV_SUM_MODE)),
            "spectral_entrywise_delta_mode": str(
                getattr(args, "spectral_entrywise_delta_mode", DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE)
            ),
        }

    svd_components_grid = getattr(args, "svd_components_grid", [20, 25, 30]) or [20, 25, 30]
    return {
        "component_grid": list(svd_components_grid),
        "n_components": getattr(args, "svd_n_components", None),
        "block_size": int(getattr(args, "stream_block_size", 131072)),
        "dtype": str(getattr(args, "dtype", "float32")),
        "acceptance_spearman_threshold": float(getattr(args, "acceptance_spearman_threshold", 0.99)),
        "acceptance_variance_threshold": float(getattr(args, "acceptance_variance_threshold", 0.95)),
        "run_offline_label_diagnostics": bool(getattr(args, "run_offline_label_diagnostics", False)),
    }


def _cmd_run_clustering(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_clustering")
    result = run_clustering_pipeline(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        feature_file=args.feature_file,
        extractor_name=args.extractor,
        extractor_params=_extractor_params_from_args(args),
        output_root=output_root,
        run_id=args.run_id,
        algorithms=args.algorithms,
        k_list=args.k_list,
        eps_list=args.eps_list,
        min_samples=args.min_samples,
        gmm_components=args.gmm_components,
        gmm_covariance_types=args.gmm_covariance_types,
        selection_metric=args.selection_metric,
        use_offline_label_metrics=args.use_offline_label_metrics,
        stability_seeds=args.stability_seeds,
        score_percentiles=args.score_percentiles,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    if result.get("score_table"):
        print(f"Score table: {result['score_table']}")
    return 0


def _cmd_run_gmm_train_inference(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_gmm_train_inference")
    result = run_gmm_train_inference_pipeline(
        manifest_json=args.manifest_json,
        dataset_root=args.dataset_root,
        output_root=output_root,
        run_id=args.run_id,
        svd_components_grid=args.svd_components_grid,
        gmm_components=args.gmm_components,
        gmm_covariance_types=args.gmm_covariance_types,
        stability_seeds=args.stability_seeds,
        score_percentiles=args.score_percentiles,
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    print(f"Train scores: {result['train_scores_csv']}")
    print(f"Inference scores: {result['inference_scores_csv']}")
    return 0


def _cmd_run_unsupervised_tsne(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_unsupervised_tsne")
    result = run_unsupervised_tsne_pipeline(
        feature_file=args.feature_file,
        output_root=output_root,
        run_id=args.run_id,
        feature_root=args.feature_root,
        model_names_file=args.feature_model_names_file,
        labels_file=args.feature_labels_file,
        metadata_file=args.feature_metadata_file,
        dataset_reference_report=args.dataset_reference_report,
        over=args.over,
        view=args.view,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        metric=args.metric,
        init=args.init,
        random_state=args.random_state,
        perplexities=args.perplexities,
        learning_rates=args.learning_rates,
        max_iters=args.max_iters_grid,
        metrics=args.metrics,
        inits=args.inits,
        random_states=args.random_states,
        standardize=args.standardize,
        point_size=args.point_size,
        alpha=args.alpha,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    print(f"Plots: {result['plot_dir']}")
    print(f"Embeddings: {result['embedding_dir']}")
    return 0


def _cmd_run_unsupervised_cnn_tsne(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_unsupervised_cnn_tsne")
    result = run_supervised_cnn_feature_tsne_pipeline(
        run_dir=args.run_dir,
        output_root=output_root,
        run_id=args.run_id,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        metric=args.metric,
        init=args.init,
        random_state=args.random_state,
        standardize=args.standardize,
        point_size=args.point_size,
        alpha=args.alpha,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    print(f"Plots: {result['plot_dir']}")
    print(f"Embeddings: {result['embedding_dir']}")
    print(f"Rows: {result['row_metadata']}")
    return 0


def _cmd_run_unsupervised_layer_scatter(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_unsupervised_layer_scatter")
    result = run_unsupervised_layer_scatter_pipeline(
        feature_file=args.feature_file,
        output_root=output_root,
        run_id=args.run_id,
        feature_root=args.feature_root,
        model_names_file=args.feature_model_names_file,
        labels_file=args.feature_labels_file,
        metadata_file=args.feature_metadata_file,
        dataset_reference_report=args.dataset_reference_report,
        point_size=args.point_size,
        alpha=args.alpha,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    print(f"Plots: {result['plot_dir']}")
    return 0


def _cmd_run_unsupervised_rank_feature_values(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_unsupervised_rank_feature_values")
    result = run_unsupervised_rank_feature_values_pipeline(
        feature_file=args.feature_file,
        output_root=output_root,
        run_id=args.run_id,
        feature_root=args.feature_root,
        model_names_file=args.feature_model_names_file,
        labels_file=args.feature_labels_file,
        metadata_file=args.feature_metadata_file,
        dataset_reference_report=args.dataset_reference_report,
        requested_features=args.features,
        point_size=args.point_size,
        alpha=args.alpha,
        jitter=args.jitter,
    )

    print("Run complete")
    print(f"Run dir: {result['run_dir']}")
    print(f"Report: {result['report']}")
    print(f"Plots: {result['plot_dir']}")
    print(f"Sample table: {result['sample_table']}")
    return 0


def _cmd_run_supervised(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root, "run_supervised")
    result = run_supervised_pipeline(
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
        stream_block_size=args.stream_block_size,
        dtype_name=args.dtype,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        train_split_percent=args.train_split,
        calibration_split_percent=args.calibration_split,
        accepted_fpr=args.accepted_fpr,
        split_by_folder=args.split_by_folder,
        cv_random_states=args.cv_seeds,
        n_jobs=args.n_jobs,
        score_percentiles=args.score_percentiles,
        tuning_executor=args.tuning_executor,
        slurm_partition=args.slurm_partition,
        slurm_max_concurrent=args.slurm_max_concurrent,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        finalize_export_shards=args.finalize_export_shards,
        stage=args.stage,
        run_dir=args.run_dir,
        task_index=args.task_index,
        task_mode=args.task_mode,
        multiclass_attack_names=(
            list(args.multiclass_attack_names) if args.multiclass_attack_names is not None else None
        ),
        cnn_hyperparams=args.cnn_hyperparams,
        autoencoder_hyperparams=args.autoencoder_hyperparams,
        few_shot_hyperparams=args.few_shot_hyperparams,
        dann_source_rank=args.dann_source_rank,
        dann_target_adaptation_percent=args.dann_target_adaptation_percent,
        dann_lambda_max=args.dann_lambda_max,
        dann_lambda_gamma=args.dann_lambda_gamma,
        dann_lr_alpha=args.dann_lr_alpha,
        dann_lr_beta=args.dann_lr_beta,
        class_weight_loss=args.class_weight_loss,
        rank_label_weight_loss=args.rank_label_weight_loss,
        skip_feature_importance=args.skip_feature_importance,
        selection_metric=args.selection_metric,
        feature_file=args.feature_file,
    )

    print("Run complete")
    if result.get("run_dir"):
        print(f"Run dir: {result['run_dir']}")
    if result.get("tuning_manifest"):
        print(f"Tuning manifest: {result['tuning_manifest']}")
    if result.get("result_path"):
        print(f"Task result: {result['result_path']}")
    if result.get("report"):
        print(f"Report: {result['report']}")
    if result.get("train_scores_csv"):
        print(f"Train scores: {result['train_scores_csv']}")
    if result.get("inference_scores_csv"):
        print(f"Inference scores: {result['inference_scores_csv']}")
    if result.get("next_steps"):
        print("Next steps:")
        for step in result["next_steps"]:
            print(f"- {step}")
    return 0


def _cmd_util_aggregate_features(args: argparse.Namespace) -> int:
    outputs = aggregate_features(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        operator=args.operator,
        features=args.features,
        spectral_qv_sum_mode=args.spectral_qv_sum_mode,
        layout=args.layout,
    )

    print("Feature aggregation complete")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    if outputs["group_mask_path"] is not None:
        print(f"Group mask: {outputs['group_mask_path']}")
    if outputs["value_mask_path"] is not None:
        print(f"Value mask: {outputs['value_mask_path']}")
    if outputs["group_names_path"] is not None:
        print(f"Group names: {outputs['group_names_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    print(f"Aggregation report: {outputs['aggregation_report_path']}")
    return 0


def _cmd_feature_extract(args: argparse.Namespace) -> int:
    start = perf_counter()
    output_root = _resolve_output_root(args.output_root, "feature_extract")
    manifest_json = resolve_manifest_path(args.manifest_json)
    items = parse_single_manifest_json(
        manifest_path=manifest_json,
        dataset_root=args.dataset_root,
        section_key="path",
    )

    ctx = create_run_context(
        pipeline="feature_extract",
        output_root=output_root,
        run_id=args.run_id,
    )

    bundle, artifacts, warnings = extract_features(
        extractor_name=args.extractor,
        items=items,
        params=_extractor_params_from_args(args),
        run_features_dir=ctx.features_dir,
    )
    kept_model_names = set(bundle.model_names)
    kept_items = [item for item in items if item.model_name in kept_model_names]
    elapsed_seconds = float(perf_counter() - start)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "extractor": args.extractor,
        "feature_shape": [int(bundle.features.shape[0]), int(bundle.features.shape[1])],
        "labels_available": bool(bundle.labels is not None),
        "model_count": len(bundle.model_names),
        "warnings": warnings,
        "metadata": bundle.metadata,
    }
    dataset_reference_payload = build_dataset_reference_payload_from_items(
        items=kept_items,
        artifact_kind="feature_extract",
        manifest_json=manifest_json,
        dataset_root=args.dataset_root,
        artifact_model_count=len(bundle.model_names),
        source_artifacts=[str(manifest_json)],
    )
    dataset_reference_report_path = write_dataset_reference_report(
        default_dataset_reference_report_path(ctx.reports_dir),
        dataset_reference_payload,
    )
    report["dataset_reference_report_path"] = str(dataset_reference_report_path)
    report_path = ctx.reports_dir / "feature_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2)

    ctx.add_artifact("features", Path(artifacts["feature_path"]))
    if artifacts.get("labels_path"):
        ctx.add_artifact("labels", Path(str(artifacts["labels_path"])))
    ctx.add_artifact("feature_metadata", Path(artifacts["metadata_path"]))
    ctx.add_artifact("dataset_reference_report", dataset_reference_report_path)
    ctx.add_artifact("report", report_path)
    ctx.add_timing("feature_extract_elapsed_seconds", elapsed_seconds)

    run_config = {
        "pipeline": "feature_extract",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "manifest_json": str(manifest_json),
        "dataset_root": str(args.dataset_root),
        "extractor": args.extractor,
        "extractor_params": _extractor_params_from_args(args),
        "warnings": warnings,
    }
    ctx.finalize(run_config)

    print("Feature extraction complete")
    print(f"Run dir: {ctx.run_dir}")
    print(f"Report: {report_path}")
    print(f"Dataset references: {dataset_reference_report_path}")
    return 0


def _cmd_download_dataset(args: argparse.Namespace) -> int:
    from .utilities.data import dataset_download

    passthrough = _normalize_download_args(list(args.download_args))
    passthrough = _rewrite_download_local_dir(passthrough)
    old_argv = sys.argv
    try:
        sys.argv = ["download_dataset"] + passthrough
        dataset_download.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_util_merge_features(args: argparse.Namespace) -> int:
    outputs = merge_feature_files(
        feature_paths=list(args.merge),
        output_filename=args.output_filename,
        feature_root=args.feature_root,
    )

    print("Feature merge complete")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    print(f"Merge report: {outputs['merge_report_path']}")
    return 0


def _cmd_util_export_feature_subset(args: argparse.Namespace) -> int:
    outputs = export_feature_subset(
        feature_file=args.feature_file,
        output_filename=args.output_filename,
        feature_root=args.feature_root,
        dataset_names=args.dataset_names,
        subset_names=args.subset_names,
        model_families=args.model_families,
        attack_names=args.attack_names,
        model_names=args.model_names,
        features=args.features,
    )

    print("Feature subset export complete")
    print(f"Feature file: {outputs['feature_path']}")
    print(f"Model names: {outputs['model_names_path']}")
    if outputs["labels_path"] is not None:
        print(f"Labels: {outputs['labels_path']}")
    print(f"Metadata: {outputs['metadata_path']}")
    print(f"Dataset references: {outputs['dataset_reference_report_path']}")
    print(f"Subset report: {outputs['subset_report_path']}")
    return 0


def _cmd_experiment_module(args: argparse.Namespace) -> int:
    module = import_module(str(args.experiment_module))
    result = module.main(list(args.experiment_args))
    return 0 if result is None else int(result)


def _add_experiment_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    module: str,
    help_text: str,
) -> None:
    command = subparsers.add_parser(name, help=help_text, add_help=False)
    command.add_argument("experiment_args", nargs=argparse.REMAINDER)
    command.set_defaults(
        func=_cmd_experiment_module,
        experiment_module=module,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="upeftguard unified CLI")
    sub = parser.add_subparsers(dest="command_group", required=True)

    run_parser = sub.add_parser("run", help="Run end-to-end pipelines")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)

    clustering = run_sub.add_parser("clustering", help="Run clustering pipeline")
    clustering.add_argument("--manifest-json", type=Path, default=None, help="Single-set manifest JSON")
    clustering.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help=dataset_root_help(),
    )
    clustering.add_argument("--feature-file", type=Path, default=None, help="Optional external feature matrix (.npy)")
    clustering.add_argument("--extractor", choices=supported_extractors(), default="svd", help="Feature extractor")
    clustering.add_argument("--output-root", type=Path, default=Path("runs"), help="Output root for run artifacts")
    clustering.add_argument("--run-id", type=str, default=None, help="Optional fixed run id")
    clustering.add_argument("--svd-components-grid", nargs="+", type=int, default=[20, 25, 30])
    clustering.add_argument("--svd-n-components", type=int, default=None)
    clustering.add_argument("--spectral-features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    clustering.add_argument("--spectral-sv-top-k", type=int, default=8)
    clustering.add_argument(
        "--spectral-moment-source",
        choices=["entrywise", "sv", "both"],
        default=DEFAULT_SPECTRAL_MOMENT_SOURCE,
    )
    clustering.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default=DEFAULT_SPECTRAL_QV_SUM_MODE,
    )
    clustering.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    clustering.add_argument("--stream-block-size", type=int, default=131072)
    clustering.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    clustering.add_argument("--acceptance-spearman-threshold", type=float, default=0.99)
    clustering.add_argument("--acceptance-variance-threshold", type=float, default=0.95)
    clustering.add_argument("--run-offline-label-diagnostics", action="store_true")

    clustering.add_argument(
        "--algorithms",
        nargs="+",
        default=["kmeans", "hierarchical", "dbscan", "gmm", "mahalanobis", "isolation_forest", "lof"],
    )
    clustering.add_argument("--k-list", nargs="+", type=int, default=[2, 3, 4, 5])
    clustering.add_argument("--eps-list", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    clustering.add_argument("--min-samples", type=int, default=2)
    clustering.add_argument("--gmm-components", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    clustering.add_argument(
        "--gmm-covariance-types",
        nargs="+",
        default=["diag", "full", "tied", "spherical"],
    )
    clustering.add_argument("--selection-metric", choices=["silhouette", "bic", "stability"], default="silhouette")
    clustering.add_argument("--use-offline-label-metrics", action="store_true")
    clustering.add_argument("--stability-seeds", nargs="+", type=int, default=[42, 43, 44])
    clustering.add_argument("--score-percentiles", nargs="+", type=float, default=[90, 95, 97, 99])
    clustering.set_defaults(func=_cmd_run_clustering)

    gmm = run_sub.add_parser("gmm-train-inference", help="Run SVD+GMM train/inference pipeline")
    gmm.add_argument("--manifest-json", type=Path, required=True)
    gmm.add_argument("--dataset-root", type=Path, default=default_dataset_root(), help=dataset_root_help())
    gmm.add_argument("--output-root", type=Path, default=Path("runs"))
    gmm.add_argument("--run-id", type=str, default=None)
    gmm.add_argument("--svd-components-grid", nargs="+", type=int, default=[20, 25, 30])
    gmm.add_argument("--gmm-components", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    gmm.add_argument(
        "--gmm-covariance-types",
        nargs="+",
        default=["diag", "full", "tied", "spherical"],
    )
    gmm.add_argument("--stability-seeds", nargs="+", type=int, default=[42, 43, 44])
    gmm.add_argument("--score-percentiles", nargs="+", type=float, default=[90, 95, 97, 99])
    gmm.add_argument("--stream-block-size", type=int, default=131072)
    gmm.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    gmm.add_argument("--reg-covar", type=float, default=1e-5)
    gmm.add_argument("--n-init", type=int, default=1)
    gmm.set_defaults(func=_cmd_run_gmm_train_inference)

    unsupervised_tsne = run_sub.add_parser(
        "unsupervised-tsne",
        help="Run t-SNE over a feature bundle, grouped by rank",
    )
    unsupervised_tsne.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="Input feature run name or explicit spectral feature .npy file",
    )
    unsupervised_tsne.add_argument("--output-root", type=Path, default=Path("runs"))
    unsupervised_tsne.add_argument("--run-id", type=str, default=None)
    unsupervised_tsne.add_argument(
        "--feature-root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    unsupervised_tsne.add_argument(
        "--feature-model-names-file",
        type=Path,
        default=None,
        help="Optional model-names JSON for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_tsne.add_argument(
        "--feature-labels-file",
        type=Path,
        default=None,
        help="Optional labels .npy for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_tsne.add_argument(
        "--feature-metadata-file",
        type=Path,
        default=None,
        help="Optional metadata JSON for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_tsne.add_argument(
        "--dataset-reference-report",
        type=Path,
        default=None,
        help="Optional dataset_reference_report.json path for rank/label provenance",
    )
    unsupervised_tsne.add_argument("--over", choices=["rank"], default="rank")
    unsupervised_tsne.add_argument("--view", choices=["full", "per_layer"], default="full")
    unsupervised_tsne.add_argument("--perplexity", type=float, default=30.0)
    unsupervised_tsne.add_argument("--learning-rate", type=_parse_learning_rate, default="auto")
    unsupervised_tsne.add_argument("--max-iter", type=int, default=1000)
    unsupervised_tsne.add_argument("--metric", type=str, default="euclidean")
    unsupervised_tsne.add_argument("--init", choices=["pca", "random"], default="pca")
    unsupervised_tsne.add_argument("--random-state", type=int, default=42)
    unsupervised_tsne.add_argument(
        "--perplexities",
        nargs="+",
        type=float,
        default=None,
        help="Optional sweep grid for perplexity; overrides the single --perplexity value when provided",
    )
    unsupervised_tsne.add_argument(
        "--learning-rates",
        nargs="+",
        type=_parse_learning_rate,
        default=None,
        help="Optional sweep grid for learning rate; overrides the single --learning-rate value when provided",
    )
    unsupervised_tsne.add_argument(
        "--max-iters-grid",
        nargs="+",
        type=int,
        default=None,
        help="Optional sweep grid for max_iter; overrides the single --max-iter value when provided",
    )
    unsupervised_tsne.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional sweep grid for distance metric; overrides the single --metric value when provided",
    )
    unsupervised_tsne.add_argument(
        "--inits",
        nargs="+",
        choices=["pca", "random"],
        default=None,
        help="Optional sweep grid for initialization; overrides the single --init value when provided",
    )
    unsupervised_tsne.add_argument(
        "--random-states",
        nargs="+",
        type=int,
        default=None,
        help="Optional sweep grid for random_state; overrides the single --random-state value when provided",
    )
    unsupervised_tsne.add_argument("--point-size", type=float, default=28.0)
    unsupervised_tsne.add_argument("--alpha", type=float, default=0.85)
    unsupervised_tsne.add_argument(
        "--no-standardize",
        action="store_false",
        dest="standardize",
        help="Disable feature standardization before t-SNE",
    )
    unsupervised_tsne.set_defaults(func=_cmd_run_unsupervised_tsne, standardize=True)

    unsupervised_cnn_tsne = run_sub.add_parser(
        "unsupervised-cnn-tsne",
        help="Run t-SNE over a finalized supervised CNN feature extractor",
    )
    unsupervised_cnn_tsne.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed supervised CNN run directory containing reports/tuning_manifest.json and models/best_model.pt",
    )
    unsupervised_cnn_tsne.add_argument("--output-root", type=Path, default=Path("runs"))
    unsupervised_cnn_tsne.add_argument("--run-id", type=str, default=None)
    unsupervised_cnn_tsne.add_argument("--perplexity", type=float, default=30.0)
    unsupervised_cnn_tsne.add_argument("--learning-rate", type=_parse_learning_rate, default="auto")
    unsupervised_cnn_tsne.add_argument("--max-iter", type=int, default=1000)
    unsupervised_cnn_tsne.add_argument("--metric", type=str, default="euclidean")
    unsupervised_cnn_tsne.add_argument("--init", choices=["pca", "random"], default="pca")
    unsupervised_cnn_tsne.add_argument("--random-state", type=int, default=None)
    unsupervised_cnn_tsne.add_argument("--point-size", type=float, default=28.0)
    unsupervised_cnn_tsne.add_argument("--alpha", type=float, default=0.85)
    unsupervised_cnn_tsne.add_argument(
        "--no-standardize",
        action="store_false",
        dest="standardize",
        help="Disable feature standardization before t-SNE",
    )
    unsupervised_cnn_tsne.set_defaults(func=_cmd_run_unsupervised_cnn_tsne, standardize=True)

    unsupervised_layer_scatter = run_sub.add_parser(
        "unsupervised-layer-scatter",
        help="Save per-feature layer-vs-value scatter and box plots for a feature bundle",
    )
    unsupervised_layer_scatter.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="Input feature run name or explicit spectral feature .npy file",
    )
    unsupervised_layer_scatter.add_argument("--output-root", type=Path, default=Path("runs"))
    unsupervised_layer_scatter.add_argument("--run-id", type=str, default=None)
    unsupervised_layer_scatter.add_argument(
        "--feature-root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    unsupervised_layer_scatter.add_argument(
        "--feature-model-names-file",
        type=Path,
        default=None,
        help="Optional model-names JSON for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_layer_scatter.add_argument(
        "--feature-labels-file",
        type=Path,
        default=None,
        help="Optional labels .npy for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_layer_scatter.add_argument(
        "--feature-metadata-file",
        type=Path,
        default=None,
        help="Optional metadata JSON for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_layer_scatter.add_argument(
        "--dataset-reference-report",
        type=Path,
        default=None,
        help="Optional dataset_reference_report.json path for rank/label provenance",
    )
    unsupervised_layer_scatter.add_argument("--point-size", type=float, default=6.0)
    unsupervised_layer_scatter.add_argument("--alpha", type=float, default=0.18)
    unsupervised_layer_scatter.set_defaults(func=_cmd_run_unsupervised_layer_scatter)

    unsupervised_rank_feature_values = run_sub.add_parser(
        "unsupervised-rank-feature-values",
        help="Save per-sample per-feature value plots across ranks after averaging each sample across layers",
    )
    unsupervised_rank_feature_values.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="Input feature run name or explicit spectral feature .npy file",
    )
    unsupervised_rank_feature_values.add_argument("--output-root", type=Path, default=Path("runs"))
    unsupervised_rank_feature_values.add_argument("--run-id", type=str, default=None)
    unsupervised_rank_feature_values.add_argument(
        "--feature-root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    unsupervised_rank_feature_values.add_argument(
        "--feature-model-names-file",
        type=Path,
        default=None,
        help="Optional model-names JSON for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_rank_feature_values.add_argument(
        "--feature-labels-file",
        type=Path,
        default=None,
        help="Optional labels .npy for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_rank_feature_values.add_argument(
        "--feature-metadata-file",
        type=Path,
        default=None,
        help="Optional metadata JSON for --feature-file (defaults to the sibling companion file)",
    )
    unsupervised_rank_feature_values.add_argument(
        "--dataset-reference-report",
        type=Path,
        default=None,
        help="Optional dataset_reference_report.json path for rank/label provenance",
    )
    unsupervised_rank_feature_values.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Optional spectral feature groups to plot; "
            "omit or pass 'all' to include every emitted feature family"
        ),
    )
    unsupervised_rank_feature_values.add_argument("--point-size", type=float, default=10.0)
    unsupervised_rank_feature_values.add_argument("--alpha", type=float, default=0.28)
    unsupervised_rank_feature_values.add_argument("--jitter", type=float, default=0.08)
    unsupervised_rank_feature_values.set_defaults(func=_cmd_run_unsupervised_rank_feature_values)

    supervised = run_sub.add_parser("supervised", help="Run supervised spectral feature pipeline")
    supervised.add_argument("--manifest-json", type=Path, default=None)
    supervised.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help=dataset_root_help(),
    )
    supervised.add_argument("--output-root", type=Path, default=Path("runs"))
    supervised.add_argument("--run-id", type=str, default=None)
    supervised.add_argument("--model", choices=["all", *registered_models()], default="logistic_regression")
    supervised.add_argument(
        "--task-mode",
        choices=["binary", "attack_family_multiclass"],
        default="binary",
        help=(
            "Supervised label/task interpretation. 'binary' keeps the existing clean-vs-backdoor "
            "behavior; 'attack_family_multiclass' trains an exclusive class over clean plus the "
            "configured attack families."
        ),
    )
    supervised.add_argument(
        "--multiclass-attack-names",
        nargs="+",
        default=None,
        help=(
            "Required for --task-mode attack_family_multiclass. Provide the positive attack-class "
            "vocabulary in the desired output order. For open-set evaluation, include the held-out "
            "attack names here so unseen positives can be scored as unknown_attack at inference."
        ),
    )
    supervised.add_argument(
        "--cnn-hyperparams",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing the cnn_1d hyperparameter search space. Each key must map "
            "to a list. If every list has one item, cnn_1d runs with a fixed configuration; otherwise "
            "the Cartesian grid is evaluated. Defaults to "
            f"{default_cnn_hyperparams_path()}."
        ),
    )
    supervised.add_argument(
        "--autoencoder-hyperparams",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing the autoencoder_knn hyperparameter search space. Each key must map "
            "to a list. If every list has one item, autoencoder_knn runs with a fixed configuration; otherwise "
            "the Cartesian grid is evaluated. Defaults to "
            f"{default_autoencoder_hyperparams_path()}."
        ),
    )
    supervised.add_argument(
        "--few-shot-hyperparams",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing the few_shot_cnn hyperparameter search space. Each key must map "
            "to a list. If every list has one item, few_shot_cnn runs with a fixed configuration; otherwise "
            "the Cartesian grid is evaluated. Defaults to "
            f"{default_few_shot_hyperparams_path()}."
        ),
    )
    supervised.add_argument(
        "--dann-source-rank",
        type=int,
        default=DANN_DEFAULT_SOURCE_RANK,
        help="Source rank expected in the cnn_1d_dann manifest train split.",
    )
    supervised.add_argument(
        "--dann-target-adaptation-percent",
        "--dann-adaptation-percent",
        dest="dann_target_adaptation_percent",
        type=int,
        default=DANN_DEFAULT_TARGET_ADAPTATION_PERCENT,
        help=(
            "Deprecated compatibility option. cnn_1d_dann now derives seen training ranks and "
            "zero-shot inference ranks from the joint manifest train/infer partitions."
        ),
    )
    supervised.add_argument(
        "--dann-lambda-max",
        type=float,
        default=DANN_DEFAULT_LAMBDA_MAX,
        help="Maximum gradient-reversal strength for cnn_1d_dann.",
    )
    supervised.add_argument(
        "--dann-lambda-gamma",
        type=float,
        default=DANN_DEFAULT_LAMBDA_GAMMA,
        help="Logistic schedule gamma for cnn_1d_dann gradient reversal.",
    )
    supervised.add_argument(
        "--dann-lr-alpha",
        type=float,
        default=DANN_DEFAULT_LR_ALPHA,
        help="Deprecated compatibility option; cnn_1d_dann now uses a fixed learning rate.",
    )
    supervised.add_argument(
        "--dann-lr-beta",
        type=float,
        default=DANN_DEFAULT_LR_BETA,
        help="Deprecated compatibility option; cnn_1d_dann now uses a fixed learning rate.",
    )
    supervised.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Required for stage=prepare/all. Selects feature groups from the extracted spectral bundle.",
    )
    supervised.add_argument("--spectral-sv-top-k", type=int, default=8)
    supervised.add_argument(
        "--spectral-moment-source",
        choices=["entrywise", "sv", "both"],
        default=DEFAULT_SPECTRAL_MOMENT_SOURCE,
    )
    supervised.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default=DEFAULT_SPECTRAL_QV_SUM_MODE,
    )
    supervised.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    supervised.add_argument("--stream-block-size", type=int, default=131072)
    supervised.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    supervised.add_argument("--cv-folds", type=int, default=5)
    supervised.add_argument(
        "--selection-metric",
        choices=list(SUPPORTED_SELECTION_METRICS),
        default="task_default",
        help=(
            "Metric used to choose the winning hyperparameters. task_default uses roc_auc for binary "
            "tasks and macro_f1 for multiclass tasks; binary_auroc uses the clean-vs-any-attack "
            "projection even for multiclass models."
        ),
    )
    supervised.add_argument("--random-state", type=int, default=42)
    supervised.add_argument(
        "--train-split",
        "--train_split",
        dest="train_split",
        type=int,
        default=100,
        help=(
            "For single manifests only, take this percentage of each split bucket as training data "
            "(globally by class unless --split-by-folder is set)."
        ),
    )
    supervised.add_argument(
        "--split-by-folder",
        action="store_true",
        help=(
            "Apply folder/label-aware splitting instead of global label stratification. "
            "For single manifests this affects both the outer train/inference split and, when enabled, "
            "the train/calibration split; for joint manifests it applies to calibration when present. "
            "When cross-validation is used, it also stratifies CV folds by folder and label."
        ),
    )
    supervised.add_argument(
        "--class-weight-loss",
        action="store_true",
        help="For CNN models, use class-balanced weights in the supervised label loss.",
    )
    supervised.add_argument(
        "--rank-label-weight-loss",
        action="store_true",
        help=(
            "For CNN models, weight supervised label loss so each observed (rank, label) bucket "
            "contributes equally."
        ),
    )
    supervised.add_argument(
        "--calibration-split",
        "--calibration_split",
        dest="calibration_split",
        type=int,
        default=None,
        help=(
            "Take this percentage of the training partition as a calibration set for threshold selection. "
            "Must be provided together with --accepted-fpr."
        ),
    )
    supervised.add_argument(
        "--accepted-fpr",
        "--accepted_fpr",
        dest="accepted_fpr",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Select one or more calibration thresholds by maximizing recall subject to "
            "false_positive_rate <= each provided value. Must be provided together with "
            "--calibration-split."
        ),
    )
    supervised.add_argument("--cv-seeds", nargs="+", type=int, default=None)
    supervised.add_argument("--n-jobs", type=int, default=-1)
    supervised.add_argument("--score-percentiles", nargs="+", type=float, default=None)
    supervised.add_argument(
        "--tuning-executor",
        choices=["local", "slurm_array"],
        default="local",
        help=(
            "Where tuning tasks run. 'local' executes them in the current process; "
            "'slurm_array' prepares the run for distributed Slurm array workers."
        ),
    )
    supervised.add_argument("--slurm-partition", type=str, default="extra")
    supervised.add_argument("--slurm-max-concurrent", type=str, default="auto")
    supervised.add_argument("--slurm-cpus-per-task", type=str, default="auto")
    supervised.add_argument(
        "--finalize-export-shards",
        type=int,
        default=1,
        help="Number of distributed finalize shards for winner feature-weight export.",
    )
    supervised.add_argument(
        "--skip-feature-importance",
        action="store_true",
        help="Skip winner feature-importance export during finalize.",
    )
    supervised.add_argument(
        "--feature-file",
        type=Path,
        default=None,
        help=(
            "Required for stage=prepare/all. Feature bundle selector: feature run name, "
            "feature output directory, or spectral_features.npy path."
        ),
    )
    supervised.add_argument(
        "--stage",
        choices=[
            "all",
            "prepare",
            "worker",
            "finalize",
            "finalize_prepare",
            "finalize_worker",
            "finalize_merge",
        ],
        default="all",
    )
    supervised.add_argument("--run-dir", type=Path, default=None)
    supervised.add_argument("--task-index", type=int, default=None)
    supervised.set_defaults(func=_cmd_run_supervised)

    feature_parser = sub.add_parser("feature", help="Feature extraction commands")
    feature_sub = feature_parser.add_subparsers(dest="feature_command", required=True)

    extract = feature_sub.add_parser("extract", help="Extract features")
    extract.add_argument("--manifest-json", type=Path, required=True)
    extract.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help=dataset_root_help(),
    )
    extract.add_argument("--extractor", choices=supported_extractors(), default="svd")
    extract.add_argument("--output-root", type=Path, default=Path("runs"))
    extract.add_argument("--run-id", type=str, default=None)
    extract.add_argument("--svd-components-grid", nargs="+", type=int, default=[20, 25, 30])
    extract.add_argument("--svd-n-components", type=int, default=None)
    extract.add_argument("--spectral-features", nargs="+", default=list(DEFAULT_SPECTRAL_FEATURES))
    extract.add_argument("--spectral-sv-top-k", type=int, default=8)
    extract.add_argument(
        "--spectral-moment-source",
        choices=["entrywise", "sv", "both"],
        default=DEFAULT_SPECTRAL_MOMENT_SOURCE,
    )
    extract.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default=DEFAULT_SPECTRAL_QV_SUM_MODE,
    )
    extract.add_argument(
        "--spectral-entrywise-delta-mode",
        choices=list(SUPPORTED_SPECTRAL_ENTRYWISE_DELTA_MODES),
        default=DEFAULT_SPECTRAL_ENTRYWISE_DELTA_MODE,
    )
    extract.add_argument("--stream-block-size", type=int, default=131072)
    extract.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    extract.add_argument("--acceptance-spearman-threshold", type=float, default=0.99)
    extract.add_argument("--acceptance-variance-threshold", type=float, default=0.95)
    extract.add_argument("--run-offline-label-diagnostics", action="store_true")
    extract.set_defaults(func=_cmd_feature_extract)

    util_parser = sub.add_parser("util", help="Utility commands")
    util_sub = util_parser.add_subparsers(dest="util_command", required=True)

    merge_features = util_sub.add_parser(
        "merge-features",
        help="Merge two spectral feature files and their companion artifacts",
    )
    merge_features.add_argument(
        "--merge",
        type=Path,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        required=True,
        help="Two run names or explicit spectral feature .npy files to merge",
    )
    merge_features.add_argument(
        "--output-filename",
        type=Path,
        required=True,
        help="Output run name or explicit output feature matrix path (.npy)",
    )
    merge_features.add_argument(
        "--feature-root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Base directory used to resolve bare run names (default: runs/feature_extract)",
    )
    merge_features.set_defaults(func=_cmd_util_merge_features)

    export_feature_subset = util_sub.add_parser(
        "export-feature-subset",
        help="Export a provenance-backed subset of a spectral feature file",
    )
    export_feature_subset.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="Input run name or explicit spectral feature .npy file",
    )
    export_feature_subset.add_argument(
        "--output-filename",
        type=Path,
        required=True,
        help="Output run name or explicit output feature matrix path (.npy)",
    )
    export_feature_subset.add_argument(
        "--feature-root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    export_feature_subset.add_argument("--dataset-name", dest="dataset_names", nargs="+", default=None)
    export_feature_subset.add_argument("--subset-name", dest="subset_names", nargs="+", default=None)
    export_feature_subset.add_argument("--model-family", dest="model_families", nargs="+", default=None)
    export_feature_subset.add_argument("--attack-name", dest="attack_names", nargs="+", default=None)
    export_feature_subset.add_argument("--model-name", dest="model_names", nargs="+", default=None)
    export_feature_subset.add_argument(
        "--features",
        "--columns",
        dest="features",
        nargs="+",
        default=None,
        help=(
            "Spectral feature groups to keep after provenance selection across all matched blocks/layers; "
            "omit or pass 'all' to keep every available feature family"
        ),
    )
    export_feature_subset.set_defaults(func=_cmd_util_export_feature_subset)

    aggregate_feature_bundle = util_sub.add_parser(
        "aggregate-features",
        help="Aggregate a spectral feature file into an architecture-independent bundle",
    )
    aggregate_feature_bundle.add_argument(
        "--feature-file",
        type=Path,
        required=True,
        help="Input run name or explicit spectral feature .npy file",
    )
    aggregate_feature_bundle.add_argument(
        "--output-filename",
        type=Path,
        required=True,
        help="Output run name or explicit output feature matrix path (.npy)",
    )
    aggregate_feature_bundle.add_argument(
        "--operator",
        choices=["avg", "max", "min"],
        default="avg",
        help="Aggregation operator for flat layout; ignored for layer_sequence",
    )
    aggregate_feature_bundle.add_argument(
        "--layout",
        choices=["flat", "layer_sequence"],
        default="flat",
        help="Output layout: flat row-wise aggregation or padded layer-sequence tensor",
    )
    aggregate_feature_bundle.add_argument(
        "--feature-root",
        type=Path,
        default=Path("runs") / "feature_extract",
        help="Base directory used to resolve bare feature run names (default: runs/feature_extract)",
    )
    aggregate_feature_bundle.add_argument(
        "--features",
        "--columns",
        dest="features",
        nargs="+",
        default=None,
        help=(
            "Spectral feature groups to keep before aggregation; "
            "omit or pass 'all' to aggregate every available feature family"
        ),
    )
    aggregate_feature_bundle.add_argument(
        "--spectral-qv-sum-mode",
        choices=["none", "append", "only"],
        default="append",
        help="Which q+v-sum columns to keep before aggregation",
    )
    aggregate_feature_bundle.set_defaults(func=_cmd_util_aggregate_features)

    download = util_sub.add_parser("download-dataset", help="Download PADBench subsets")
    download.add_argument("download_args", nargs=argparse.REMAINDER)
    download.set_defaults(func=_cmd_download_dataset)

    experiment_parser = sub.add_parser("experiment", help="Experiment launchers and post-processing commands")
    experiment_sub = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    for command_name, (module_name, help_text) in EXPERIMENT_COMMANDS.items():
        _add_experiment_command(experiment_sub, command_name, module_name, help_text)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    if len(argv_tokens) >= 2 and argv_tokens[0] == "util" and argv_tokens[1] == "download-dataset":
        args = argparse.Namespace(
            command_group="util",
            util_command="download-dataset",
            download_args=argv_tokens[2:],
            func=_cmd_download_dataset,
        )
    elif len(argv_tokens) >= 2 and argv_tokens[0] == "experiment" and argv_tokens[1] in EXPERIMENT_COMMANDS:
        args = argparse.Namespace(
            command_group="experiment",
            experiment_command=argv_tokens[1],
            experiment_args=argv_tokens[2:],
            experiment_module=EXPERIMENT_COMMANDS[argv_tokens[1]][0],
            func=_cmd_experiment_module,
        )
    else:
        args = parser.parse_args(argv_tokens)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

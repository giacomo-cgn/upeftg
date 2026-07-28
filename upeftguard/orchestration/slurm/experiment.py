"""Slurm submission backend for experiment workflow stages."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from .client import project_root, read_dependency_job_id, run_sbatch_command
from .resources import resolve_partition_cpus


def submit_feature_extraction(
    *,
    run_dir: Path,
    run_id: str,
    manifest_json: Path,
    dataset_root: Path,
    output_root: Path,
    output_run_dir: Path,
    features: list[str],
    spectral_sv_top_k: int,
    spectral_moment_source: str,
    spectral_qv_sum_mode: str,
    spectral_entrywise_delta_mode: str,
    spectral_attention_granularity: str,
    stream_block_size: int,
    dtype_name: str,
    partition: str,
    nodes: str,
    cpus_per_worker: str,
    workers_per_node: str,
    wait_for_job_index: bool,
    parallelization_settings: list[str] | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Write the extraction controller config and submit its prepare job."""

    from .extraction import extraction_controller_command, write_extraction_slurm_config

    run_dir = Path(run_dir)
    work_dir = run_dir / ".work"
    log_dir = run_dir / "logs"
    job_index_path = work_dir / "job_index.json"
    config_path = write_extraction_slurm_config(
        work_dir / "slurm.json",
        extraction_kwargs={
            "manifest_json": Path(manifest_json),
            "dataset_root": Path(dataset_root),
            "output_root": Path(output_root),
            "output_run_dir": Path(output_run_dir),
            "run_id": str(run_id),
            "features": list(features),
            "spectral_sv_top_k": int(spectral_sv_top_k),
            "spectral_moment_source": spectral_moment_source,
            "spectral_qv_sum_mode": spectral_qv_sum_mode,
            "spectral_entrywise_delta_mode": spectral_entrywise_delta_mode,
            "spectral_attention_granularity": spectral_attention_granularity,
            "stream_block_size": int(stream_block_size),
            "dtype_name": dtype_name,
            "parallelization_settings": list(parallelization_settings or []),
        },
        partition=partition,
        nodes=nodes,
        cpus_per_worker=cpus_per_worker,
        workers_per_node=workers_per_node,
        log_dir=log_dir,
        job_index_path=job_index_path,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    submission = run_sbatch_command(
        extraction_controller_command(config_path),
        partition=partition,
        cpus_per_task=2,
        memory="8GB",
        job_name=f"upeftguard_feature_extract_prepare_{run_id}",
        output=log_dir / "feature_extract_prepare_%j.out",
        error=log_dir / "feature_extract_prepare_%j.err",
        chdir=project_root(),
        wait=wait_for_job_index,
        dry_run=dry_run,
    )
    submission["config"] = str(config_path)
    submission["job_index"] = str(job_index_path)
    return submission


def submit_feature_aggregation(
    *,
    run_dir: Path,
    run_id: str,
    feature_file: Path,
    output_filename: Path,
    output_root: Path,
    feature_root: Path,
    features: list[str],
    spectral_qv_sum_mode: str,
    spectral_attention_granularity: str | None,
    partition: str,
    cpus_per_worker: str,
    dependency: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit the single-job feature aggregation stage."""

    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "experiment",
        "aggregate",
        "--backend",
        "local",
        "--feature-file",
        str(feature_file),
        "--output-filename",
        str(output_filename),
        "--output-root",
        str(output_root),
        "--run-id",
        str(run_id),
        "--feature-root",
        str(feature_root),
        "--spectral-qv-sum-mode",
        spectral_qv_sum_mode,
    ]
    if spectral_attention_granularity is not None:
        command.extend(["--spectral-attention-granularity", spectral_attention_granularity])
    if features:
        command.extend(["--features", *features])
    return run_sbatch_command(
        command,
        partition=partition,
        dependency=dependency,
        cpus_per_task=resolve_partition_cpus(partition, cpus_per_worker),
        memory="16GB",
        job_name=f"upeftguard_aggregation_{run_id}",
        output=log_dir / "aggregation_%j.out",
        error=log_dir / "aggregation_%j.err",
        chdir=project_root(),
        dry_run=dry_run,
    )


def submit_checkpoint_inference(
    *,
    run_dir: Path,
    run_id: str,
    checkpoint: Path,
    manifest_json: Path,
    feature_file: Path,
    output_root: Path,
    partition: str,
    cpus_per_worker: str,
    dependency: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit the single-job checkpoint inference stage."""

    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "experiment",
        "infer",
        "--backend",
        "local",
        "--output-root",
        str(output_root),
        "--run-id",
        str(run_id),
    ]
    command.extend(
        [
            "--checkpoint",
            str(checkpoint),
            "--manifest-json",
            str(manifest_json),
            "--feature-file",
            str(feature_file),
        ]
    )
    return run_sbatch_command(
        command,
        partition=partition,
        dependency=dependency,
        cpus_per_task=resolve_partition_cpus(partition, cpus_per_worker),
        memory="16GB",
        job_name=f"upeftguard_inference_{run_id}",
        output=log_dir / "inference_%j.out",
        error=log_dir / "inference_%j.err",
        chdir=project_root(),
        dry_run=dry_run,
    )


def submit_grouped_explanation(
    *,
    run_dir: Path,
    run_id: str,
    checkpoint: Path,
    feature_file: Path,
    background_selection: Path,
    explanation_selection: Path,
    output_root: Path,
    partition: str,
    cpus_per_worker: str,
    dependency: str | None,
    dry_run: bool,
    background_label: str,
    background_max_samples: int | None,
    explanation_max_samples: int | None,
    random_state: int,
    batch_size: int,
    grouping: list[str] | tuple[str, ...] | None,
    plot_max_display: int,
    max_local_plots: int,
) -> dict[str, Any]:
    """Submit grouped SHAP as a compute job; the frontend only constructs argv."""

    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "upeftguard.cli",
        "run",
        "explanation-worker",
        "--checkpoint",
        str(checkpoint),
        "--feature-file",
        str(feature_file),
        "--background-selection",
        str(background_selection),
        "--explanation-selection",
        str(explanation_selection),
        "--output-root",
        str(output_root),
        "--output-run-dir",
        str(run_dir),
        "--run-id",
        str(run_id),
        "--background-label",
        str(background_label),
        "--random-state",
        str(int(random_state)),
        "--batch-size",
        str(int(batch_size)),
        "--plot-max-display",
        str(int(plot_max_display)),
        "--max-local-plots",
        str(int(max_local_plots)),
    ]
    if grouping:
        command.extend(["--grouping", *(str(value) for value in grouping)])
    if background_max_samples is not None:
        command.extend(["--background-max-samples", str(int(background_max_samples))])
    if explanation_max_samples is not None:
        command.extend(["--explanation-max-samples", str(int(explanation_max_samples))])
    return run_sbatch_command(
        command,
        partition=partition,
        dependency=dependency,
        cpus_per_task=resolve_partition_cpus(partition, cpus_per_worker),
        memory="32GB",
        job_name=f"upeftguard_explanation_{run_id}",
        output=log_dir / "explanation_%j.out",
        error=log_dir / "explanation_%j.err",
        chdir=project_root(),
        dry_run=dry_run,
    )


def submit_supervised_training(
    *,
    run_dir: Path,
    run_id: str,
    pipeline_kwargs: dict[str, Any],
    partition: str,
    nodes: str,
    cpus_per_worker: str,
    workers_per_node: str,
    dependency: str | None,
    dry_run: bool,
    wait_for_job_index: bool = False,
) -> dict[str, Any]:
    """Write the supervised controller config and submit its prepare job."""

    from .supervised import controller_command, write_supervised_slurm_config

    scheduler_kwargs = dict(pipeline_kwargs)
    scheduler_kwargs.update(
        {
            "stage": "prepare",
            "tuning_executor": "slurm_packed",
            "slurm_partition": partition,
            "cpus_per_worker": cpus_per_worker,
        }
    )
    run_dir = Path(run_dir)
    work_dir = run_dir / ".work"
    log_dir = run_dir / "logs"
    config_path = write_supervised_slurm_config(
        work_dir / "slurm.json",
        pipeline_kwargs=scheduler_kwargs,
        partition=partition,
        nodes=nodes,
        cpus_per_worker=cpus_per_worker,
        workers_per_node=workers_per_node,
        log_dir=log_dir,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    submission = run_sbatch_command(
        controller_command(config_path),
        partition=partition,
        dependency=dependency,
        cpus_per_task=4,
        memory="16GB",
        job_name=f"upeftguard_supervised_prepare_{run_id}",
        output=log_dir / "supervised_prepare_%j.out",
        error=log_dir / "supervised_prepare_%j.err",
        chdir=project_root(),
        wait=wait_for_job_index,
        dry_run=dry_run,
    )
    submission["config"] = str(config_path)
    submission["job_index"] = str(Path(pipeline_kwargs["prepared_run_dir"]) / ".work" / "job_index.json")
    return submission


def afterok_dependency(
    submission: dict[str, Any],
    *,
    job_index_path: Path | None = None,
    dry_run_placeholder: str | None = None,
) -> str | None:
    """Resolve a completed-stage submission to an ``afterok`` dependency."""

    job_id = dry_run_placeholder if submission.get("dry_run") else None
    if job_id is None and job_index_path is not None:
        job_id = read_dependency_job_id(Path(job_index_path))
    if job_id is None:
        value = submission.get("job_id")
        job_id = str(value) if value else None
    return f"afterok:{job_id}" if job_id else None

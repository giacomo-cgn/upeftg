"""Slurm-only public workflow for grouped model explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..orchestration.slurm.experiment import submit_grouped_explanation
from .common import timing_finish, timing_start
from .config import stage_context


def run_grouped_explanation(
    *,
    checkpoint: Path,
    feature_file: Path,
    background_selection: Path,
    explanation_selection: Path,
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    partition: str = "extra",
    cpus_per_worker: str = "4",
    dry_run: bool = False,
    dependency: str | None = None,
    background_label: str = "all",
    background_max_samples: int | None = None,
    explanation_max_samples: int | None = None,
    random_state: int = 42,
    batch_size: int = 128,
    grouping: list[str] | tuple[str, ...] | None = None,
    plot_max_display: int = 20,
    max_local_plots: int = 20,
) -> dict[str, Any]:
    """Submit grouped SHAP work without loading model or feature arrays on the frontend."""

    output_root = Path(output_root).expanduser().resolve()
    experiment, ctx = stage_context(
        output_root,
        run_id=run_id,
        workflow="explain",
        stage="explanation",
        backend="slurm",
    )
    experiment.update(stage="explanation", stage_status="running")
    timing, started_perf = timing_start("explanation_submit", backend="slurm")
    submission = submit_grouped_explanation(
        run_dir=ctx.run_dir,
        run_id=ctx.run_id,
        checkpoint=checkpoint,
        feature_file=feature_file,
        background_selection=background_selection,
        explanation_selection=explanation_selection,
        output_root=output_root,
        partition=partition,
        cpus_per_worker=cpus_per_worker,
        dependency=dependency,
        dry_run=dry_run,
        background_label=background_label,
        background_max_samples=background_max_samples,
        explanation_max_samples=explanation_max_samples,
        random_state=random_state,
        batch_size=batch_size,
        grouping=grouping,
        plot_max_display=plot_max_display,
        max_local_plots=max_local_plots,
    )
    timing = timing_finish(timing, started_perf)
    experiment.update(
        stage="explanation",
        stage_status="submitted" if not dry_run else "planned",
        stage_values={
            "job_ids": [submission.get("job_id")] if submission.get("job_id") else [],
            "timing": timing,
            "dependency": dependency,
            "checkpoint": experiment.display_path(Path(checkpoint)),
            "input_feature_path": experiment.display_path(Path(feature_file)),
            "background_selection": experiment.display_path(Path(background_selection)),
            "explanation_selection": experiment.display_path(Path(explanation_selection)),
        },
    )
    return {
        "run_dir": str(experiment.run_dir),
        "stage_run_dir": str(ctx.run_dir),
        "run_id": ctx.run_id,
        "backend": "slurm",
        "slurm": submission,
        "timings": {"submit": timing},
    }

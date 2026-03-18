"""Data processing pipelines.

Provides a composable pipeline framework for EEG preprocessing,
modeled after the two-pass approach in eb_jepa_eeg's preprocess_hbn.py.

Pipelines are built from ProcessingStep objects that can be chained
together. Each step has checkpoint/resume support so long-running
preprocessing jobs can be restarted safely.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a processing step execution."""

    step_name: str
    status: str  # "success", "skipped", "failed"
    n_processed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessingStep(ABC):
    """Abstract base class for a processing pipeline step.

    Subclass this to implement custom preprocessing operations.
    Each step operates on a directory of files and writes output
    to another directory.

    Example:
        class BandpassFilter(ProcessingStep):
            name = "bandpass_filter"

            def __init__(self, l_freq: float, h_freq: float):
                self.l_freq = l_freq
                self.h_freq = h_freq

            def process_file(self, input_path, output_path, **kwargs):
                raw = mne.io.read_raw(input_path, preload=True)
                raw.filter(self.l_freq, self.h_freq)
                raw.save(output_path, overwrite=True)
    """

    name: str = "unnamed_step"

    @abstractmethod
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process a single file.

        Args:
            input_path: Path to the input file.
            output_path: Path where the output should be written.
            **kwargs: Additional parameters from the pipeline.

        Returns:
            Dict of metadata about the processing (stats, etc.).
        """
        ...

    def get_output_filename(self, input_path: Path) -> str:
        """Determine the output filename for a given input.

        Override this if the output format differs from the input.
        """
        return input_path.name

    def should_skip(self, input_path: Path, output_path: Path) -> bool:
        """Check if this file has already been processed (for resume support).

        Default: skip if output file exists and is newer than input.
        """
        if not output_path.exists():
            return False
        return output_path.stat().st_mtime >= input_path.stat().st_mtime


class ProcessingPipeline:
    """A composable data processing pipeline with checkpoint/resume.

    Runs a sequence of ProcessingSteps over a directory of files,
    with support for:
    - Checkpoint/resume (skip already-processed files)
    - Parallel processing (via joblib)
    - Progress logging
    - Error collection without stopping the pipeline

    Usage:
        from neurolab.data.processing import ProcessingPipeline
        from my_steps import Resample, BandpassFilter, ZScoreNormalize

        pipeline = ProcessingPipeline(
            steps=[
                Resample(target_sfreq=200.0),
                BandpassFilter(l_freq=0.5, h_freq=99.5),
                ZScoreNormalize(clip_std=15.0),
            ],
        )

        results = pipeline.run(
            input_dir="/data/raw/hbn/R1",
            output_dir="/data/preprocessed/hbn/R1",
            file_pattern="*.fif",
            n_jobs=8,
        )
    """

    def __init__(
        self,
        steps: list[ProcessingStep],
        checkpoint_file: str = ".pipeline_checkpoint.json",
    ) -> None:
        self.steps = steps
        self.checkpoint_file = checkpoint_file

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        file_pattern: str = "*",
        n_jobs: int = 1,
        resume: bool = True,
        **kwargs: Any,
    ) -> list[StepResult]:
        """Run the full pipeline.

        Args:
            input_dir: Directory containing input files.
            output_dir: Base output directory. Each step creates a subdirectory.
            file_pattern: Glob pattern to select input files.
            n_jobs: Number of parallel workers (1 = sequential).
            resume: If True, skip already-processed files.
            **kwargs: Additional arguments passed to each step.

        Returns:
            List of StepResult objects, one per step.
        """
        input_path = Path(input_dir)
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        results: list[StepResult] = []
        current_input = input_path

        for i, step in enumerate(self.steps):
            step_output = output_base / f"step_{i:02d}_{step.name}"
            step_output.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Pipeline step {i + 1}/{len(self.steps)}: {step.name} "
                f"({current_input} → {step_output})"
            )

            result = self._run_step(
                step=step,
                input_dir=current_input,
                output_dir=step_output,
                file_pattern=file_pattern if i == 0 else "*",
                n_jobs=n_jobs,
                resume=resume,
                **kwargs,
            )
            results.append(result)

            if result.status == "failed" and result.n_processed == 0:
                logger.error(f"Step '{step.name}' failed completely. Stopping pipeline.")
                break

            # Next step's input is this step's output
            current_input = step_output

        self._save_checkpoint(output_base, results)
        return results

    def _run_step(
        self,
        step: ProcessingStep,
        input_dir: Path,
        output_dir: Path,
        file_pattern: str,
        n_jobs: int,
        resume: bool,
        **kwargs: Any,
    ) -> StepResult:
        """Run a single processing step over all matching files."""
        start_time = time.time()
        input_files = sorted(input_dir.glob(file_pattern))

        if not input_files:
            logger.warning(f"No files matching '{file_pattern}' in {input_dir}")
            return StepResult(
                step_name=step.name, status="skipped", elapsed_seconds=0.0
            )

        n_processed = 0
        n_skipped = 0
        n_failed = 0
        errors: list[str] = []

        def _process_one(input_file: Path) -> tuple[str, str]:
            """Process a single file. Returns (status, error_msg)."""
            out_name = step.get_output_filename(input_file)
            out_file = output_dir / out_name

            if resume and step.should_skip(input_file, out_file):
                return "skipped", ""

            try:
                step.process_file(input_file, out_file, **kwargs)
                return "success", ""
            except Exception as e:
                return "failed", f"{input_file.name}: {e}"

        if n_jobs == 1:
            # Sequential processing
            for input_file in input_files:
                status, error = _process_one(input_file)
                if status == "success":
                    n_processed += 1
                elif status == "skipped":
                    n_skipped += 1
                else:
                    n_failed += 1
                    errors.append(error)
                    logger.warning(f"Failed: {error}")
        else:
            # Parallel processing
            try:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(_process_one)(f) for f in input_files
                )
                for status, error in results:
                    if status == "success":
                        n_processed += 1
                    elif status == "skipped":
                        n_skipped += 1
                    else:
                        n_failed += 1
                        errors.append(error)
            except ImportError:
                logger.warning("joblib not available, falling back to sequential")
                for input_file in input_files:
                    status, error = _process_one(input_file)
                    if status == "success":
                        n_processed += 1
                    elif status == "skipped":
                        n_skipped += 1
                    else:
                        n_failed += 1
                        errors.append(error)

        elapsed = time.time() - start_time
        overall_status = "success" if n_failed == 0 else "failed"

        logger.info(
            f"Step '{step.name}': {n_processed} processed, "
            f"{n_skipped} skipped, {n_failed} failed ({elapsed:.1f}s)"
        )

        return StepResult(
            step_name=step.name,
            status=overall_status,
            n_processed=n_processed,
            n_skipped=n_skipped,
            n_failed=n_failed,
            elapsed_seconds=elapsed,
            errors=errors,
        )

    def _save_checkpoint(self, output_dir: Path, results: list[StepResult]) -> None:
        """Save a checkpoint file with pipeline execution metadata."""
        checkpoint = {
            "steps": [
                {
                    "name": r.step_name,
                    "status": r.status,
                    "n_processed": r.n_processed,
                    "n_skipped": r.n_skipped,
                    "n_failed": r.n_failed,
                    "elapsed_seconds": r.elapsed_seconds,
                    "errors": r.errors,
                }
                for r in results
            ]
        }
        path = output_dir / self.checkpoint_file
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

"""Example: Building a preprocessing pipeline with neurolab-data.

Demonstrates how to create a multi-step EEG preprocessing pipeline
modeled after eb_jepa_eeg's preprocess_hbn.py two-pass approach,
then submit it as a SLURM job.
"""

from pathlib import Path

import numpy as np

from neurolab.data import resolve_data_path, resolve_scratch_path
from neurolab.data.processing import ProcessingStep, ProcessingPipeline
from neurolab.jobs import SlurmJob, submit_job


# ── Define processing steps ─────────────────────────────────────────

class ResampleAndFilter(ProcessingStep):
    """Pass 1: Resample and bandpass filter raw EEG recordings.

    Follows the REVE/Défossez et al. 2023 preprocessing approach.
    """

    name = "resample_filter"

    def __init__(self, target_sfreq: float = 200.0, l_freq: float = 0.5, h_freq: float = 99.5):
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq

    def process_file(self, input_path: Path, output_path: Path, **kwargs):
        import mne

        raw = mne.io.read_raw_fif(str(input_path), preload=True, verbose=False)
        raw.resample(self.target_sfreq)
        raw.filter(self.l_freq, self.h_freq, verbose=False)
        raw.save(str(output_path), overwrite=True, verbose=False)
        return {"n_samples": raw.n_times, "sfreq": raw.info["sfreq"]}


class ZScoreNormalize(ProcessingStep):
    """Pass 2: Per-channel z-score normalization with clipping.

    Computes stats across all files first, then normalizes.
    """

    name = "zscore_normalize"

    def __init__(self, clip_std: float = 15.0):
        self.clip_std = clip_std
        self._channel_means: np.ndarray | None = None
        self._channel_stds: np.ndarray | None = None

    def process_file(self, input_path: Path, output_path: Path, **kwargs):
        import mne

        raw = mne.io.read_raw_fif(str(input_path), preload=True, verbose=False)
        data = raw.get_data()  # (n_channels, n_times)

        # Per-channel normalization
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)

        normalized = (data - mean) / std
        if self.clip_std > 0:
            normalized = np.clip(normalized, -self.clip_std, self.clip_std)

        raw._data = normalized.astype(np.float32)
        raw.save(str(output_path), overwrite=True, verbose=False)
        return {"clip_std": self.clip_std}


def main():
    # ── Build the pipeline ──────────────────────────────────────────
    pipeline = ProcessingPipeline(
        steps=[
            ResampleAndFilter(target_sfreq=200.0, l_freq=0.5, h_freq=99.5),
            ZScoreNormalize(clip_std=15.0),
        ],
    )

    # ── Run locally for testing ─────────────────────────────────────
    input_dir = resolve_data_path("hbn/raw/R1/ThePresent")
    output_dir = resolve_data_path("hbn/preprocessed/R1/ThePresent", mkdir=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Uncomment to run:
    # results = pipeline.run(
    #     input_dir=input_dir,
    #     output_dir=output_dir,
    #     file_pattern="*.fif",
    #     n_jobs=8,
    #     resume=True,
    # )
    # for r in results:
    #     print(f"  {r.step_name}: {r.status} ({r.n_processed} processed, {r.elapsed_seconds:.1f}s)")

    # ── Submit as SLURM job on Expanse ──────────────────────────────
    job = SlurmJob(
        name="preprocess_hbn_R1",
        command=(
            "python -c \""
            "from examples.preprocessing_pipeline import main; main()"
            "\""
        ),
        cluster="expanse",
        time_limit="08:00:00",
        gpus=0,  # CPU-only preprocessing
    )

    script_path = submit_job(job, dry_run=True, script_dir="slurm/generated")
    print(f"\nGenerated SLURM script: {script_path}")


if __name__ == "__main__":
    main()

"""Tests for neurolab-jobs."""

import os
import tempfile
from pathlib import Path

import pytest

from neurolab.jobs import SlurmJob, SweepConfig, generate_sweep_script
from neurolab.jobs.submit import _render_job_script


def test_slurm_job_render():
    """Job rendering should produce valid SLURM script."""
    job = SlurmJob(
        name="test_train",
        command="python train.py --lr 1e-4",
        cluster="expanse",
        time_limit="04:00:00",
        gpus=1,
        working_dir="/expanse/projects/nemar/adapter_finetuning",
    )

    script = _render_job_script(job)

    assert "#!/usr/bin/env bash" in script
    assert "#SBATCH --job-name=test_train" in script
    assert "#SBATCH --partition=gpu-shared" in script
    assert "#SBATCH --account=csd403" in script
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "module purge" in script
    assert "module load gpu" in script
    assert "python train.py --lr 1e-4" in script


def test_slurm_job_default_time():
    """Should use cluster default time if not specified."""
    job = SlurmJob(
        name="test",
        command="echo hello",
        cluster="expanse",
    )
    script = _render_job_script(job)
    assert "#SBATCH --time=02:00:00" in script


def test_slurm_job_array():
    """Array spec should be included in script."""
    job = SlurmJob(
        name="test_array",
        command="echo $SLURM_ARRAY_TASK_ID",
        cluster="expanse",
        array_spec="0-23%4",
    )
    script = _render_job_script(job)
    assert "#SBATCH --array=0-23%4" in script


def test_sweep_combinations():
    """SweepConfig should generate correct number of combinations."""
    sweep = SweepConfig(
        name="test_sweep",
        command_template="python train.py adapter={adapter} model={model}",
        parameters={
            "adapter": ["lora", "ia3", "dora"],
            "model": ["labram", "eegpt"],
        },
    )
    assert sweep.n_jobs == 6
    combos = sweep.combinations
    assert len(combos) == 6
    assert {"adapter": "lora", "model": "labram"} in combos
    assert {"adapter": "dora", "model": "eegpt"} in combos


def test_sweep_script_generation():
    """Sweep script should include array spec and parameter parsing."""
    sweep = SweepConfig(
        name="test_sweep",
        command_template="python train.py adapter={adapter} model={model} data={data}",
        parameters={
            "adapter": ["lora", "ia3"],
            "model": ["labram"],
            "data": ["bcic2a", "physionet"],
        },
        cluster="expanse",
        max_concurrent=2,
    )

    script = generate_sweep_script(sweep)

    assert "#SBATCH --array=0-3%2" in script  # 4 combos, max 2
    assert "EXPERIMENTS=(" in script
    assert "IFS=',' read -r ADAPTER MODEL DATA" in script
    assert "python train.py adapter=$ADAPTER model=$MODEL data=$DATA" in script
    assert "module purge" in script


def test_sweep_single_param():
    """Sweep with one parameter should still work."""
    sweep = SweepConfig(
        name="lr_sweep",
        command_template="python train.py --lr {lr}",
        parameters={"lr": ["1e-3", "1e-4", "1e-5"]},
        cluster="expanse",
    )

    assert sweep.n_jobs == 3
    script = generate_sweep_script(sweep)
    assert "#SBATCH --array=0-2" in script


def test_job_no_slurm_cluster():
    """Submitting to a non-SLURM cluster should raise error."""
    job = SlurmJob(
        name="test",
        command="echo hello",
        cluster="local",
    )
    with pytest.raises(ValueError, match="no SLURM"):
        _render_job_script(job)

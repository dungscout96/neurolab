"""Tests for neurolab-jobs."""

import pytest

from neurolab.jobs import Job, SweepConfig
from neurolab.jobs.submit import _render_slurm_script, _render_direct_script, _build_preamble
from neurolab.jobs import get_cluster


# ── Job dataclass tests ─────────────────────────────────────────────


def test_job_required_fields():
    """Job must have name, cluster, repo_path, command."""
    job = Job(
        name="test",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/MyProject",
        command="python train.py",
    )
    assert job.name == "test"
    assert job.branch == "main"
    assert job.gpus == 1


def test_job_resolve_cluster():
    """Job should resolve its cluster config."""
    job = Job(name="test", cluster="expanse", repo_path="/tmp", command="echo hi")
    cfg = job.resolve_cluster()
    assert cfg.name == "expanse"
    assert cfg.is_hpc


# ── Preamble tests ──────────────────────────────────────────────────


def test_preamble_cd_into_repo():
    """Preamble should cd into the repo path."""
    job = Job(name="t", cluster="expanse", repo_path="/my/repo", command="echo hi")
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert "cd /my/repo" in preamble


def test_preamble_branch_checkout():
    """Preamble should checkout the specified branch."""
    job = Job(name="t", cluster="expanse", repo_path="/repo", command="echo", branch="dev")
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert "git checkout dev" in preamble
    assert "git pull --ff-only" in preamble


def test_preamble_skip_branch():
    """Empty branch should skip git checkout."""
    job = Job(name="t", cluster="expanse", repo_path="/repo", command="echo", branch="")
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert "git checkout" not in preamble


def test_preamble_venv_activation():
    """Preamble should activate the specified venv."""
    job = Job(
        name="t", cluster="expanse", repo_path="/repo", command="echo",
        venv="/my/custom/conda_env",
    )
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert 'export PATH="/my/custom/conda_env/bin:$PATH"' in preamble


def test_preamble_cluster_default_venv():
    """If no job venv, should use cluster default conda_env."""
    job = Job(name="t", cluster="expanse", repo_path="/repo", command="echo")
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert f'export PATH="{cfg.conda_env}/bin:$PATH"' in preamble


def test_preamble_skip_venv():
    """__none__ should skip venv activation."""
    job = Job(
        name="t", cluster="expanse", repo_path="/repo", command="echo",
        venv="__none__",
    )
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert "__none__" not in preamble


def test_preamble_env_vars_merged():
    """Job env_vars should override cluster defaults."""
    job = Job(
        name="t", cluster="expanse", repo_path="/repo", command="echo",
        env_vars={"WANDB_MODE": "online", "MY_VAR": "hello"},
    )
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    # Job override should win
    assert 'export WANDB_MODE="online"' in preamble
    # Job-specific var should be present
    assert 'export MY_VAR="hello"' in preamble
    # Cluster default should be present
    assert "export HF_HOME=" in preamble


def test_preamble_module_loads():
    """Preamble should include module loads for HPC clusters."""
    job = Job(name="t", cluster="expanse", repo_path="/repo", command="echo")
    cfg = get_cluster("expanse")
    preamble = _build_preamble(job, cfg)
    assert "module purge" in preamble
    assert "module load gpu" in preamble


def test_preamble_no_modules_jamming():
    """Jamming should not have module loads."""
    job = Job(name="t", cluster="jamming", repo_path="/repo", command="echo")
    cfg = get_cluster("jamming")
    preamble = _build_preamble(job, cfg)
    assert "module" not in preamble


# ── SLURM script rendering ──────────────────────────────────────────


def test_slurm_script_render():
    """SLURM script should contain all expected directives."""
    job = Job(
        name="train_run",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
        command="python scripts/train.py adapter=lora model=labram",
        venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
        time_limit="04:00:00",
        gpus=1,
    )
    cfg = get_cluster("expanse")
    script = _render_slurm_script(job, cfg)

    assert "#!/usr/bin/env bash" in script
    assert "#SBATCH --job-name=train_run" in script
    assert "#SBATCH --partition=gpu-shared" in script
    assert "#SBATCH --account=csd403" in script
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "cd /expanse/projects/nemar/dtyoung/OpenEEG-Bench" in script
    assert "git checkout main" in script
    assert 'export PATH="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning/bin:$PATH"' in script
    assert "module purge" in script
    assert "module load gpu" in script
    assert "python scripts/train.py adapter=lora model=labram" in script


def test_slurm_default_time():
    """Should use cluster default time if not specified."""
    job = Job(name="t", cluster="expanse", repo_path="/repo", command="echo hi")
    cfg = get_cluster("expanse")
    script = _render_slurm_script(job, cfg)
    assert "#SBATCH --time=02:00:00" in script


def test_slurm_array_spec():
    """Array spec should appear in script."""
    job = Job(
        name="t", cluster="expanse", repo_path="/repo", command="echo",
        array_spec="0-23%4",
    )
    cfg = get_cluster("expanse")
    script = _render_slurm_script(job, cfg)
    assert "#SBATCH --array=0-23%4" in script


def test_slurm_delta_different_from_expanse():
    """Delta should produce different partition/account than Expanse."""
    job = Job(name="t", cluster="delta", repo_path="/repo", command="echo")
    cfg = get_cluster("delta")
    script = _render_slurm_script(job, cfg)
    assert "#SBATCH --partition=gpuA100x4" in script
    assert "#SBATCH --account=bcfj-delta-gpu" in script
    assert "module load anaconda3_gpu" in script


# ── Direct (non-SLURM) script rendering ─────────────────────────────


def test_direct_script_render():
    """Direct script for jamming should have no SLURM directives."""
    job = Job(
        name="train_run",
        cluster="jamming",
        repo_path="~/projects/OpenEEG-Bench",
        command="python scripts/train.py adapter=lora",
        branch="dev",
    )
    cfg = get_cluster("jamming")
    script = _render_direct_script(job, cfg)

    assert "set -euo pipefail" in script
    assert "cd ~/projects/OpenEEG-Bench" in script
    assert "git checkout dev" in script
    assert "nohup" in script
    assert "python scripts/train.py adapter=lora" in script
    assert "disown" in script
    assert "#SBATCH" not in script


# ── SSH alias tests ──────────────────────────────────────────────────


def test_ssh_alias_expanse():
    """Expanse ssh_target should be 'expanse'."""
    cfg = get_cluster("expanse")
    assert cfg.ssh_target == "expanse"


def test_ssh_alias_delta():
    """Delta ssh_target should be 'delta'."""
    cfg = get_cluster("delta")
    assert cfg.ssh_target == "delta"


def test_ssh_alias_jamming():
    """Jamming ssh_target should fall back to login_host IP (no alias)."""
    cfg = get_cluster("jamming")
    assert cfg.ssh_target == "100.113.196.11"


def test_ssh_target_fallback():
    """ssh_target should fall back to login_host if no alias."""
    cfg = get_cluster("local")
    # local has neither, so ssh_target is empty
    assert cfg.ssh_target == ""


# ── Job.submit routing ───────────────────────────────────────────────


def test_submit_slurm_dry_run():
    """dry_run on SLURM cluster should return a script string."""
    job = Job(
        name="test",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/MyProject",
        command="python train.py",
    )
    result = job.submit(dry_run=True)
    assert "#!/usr/bin/env bash" in result
    assert "#SBATCH" in result
    assert "python train.py" in result


def test_submit_direct_dry_run():
    """dry_run on non-SLURM cluster should return a direct script."""
    job = Job(
        name="test",
        cluster="jamming",
        repo_path="~/projects/MyProject",
        command="python train.py",
    )
    result = job.submit(dry_run=True)
    assert "set -euo pipefail" in result
    assert "#SBATCH" not in result
    assert "python train.py" in result


# ── Sweep tests ──────────────────────────────────────────────────────


def test_sweep_combinations():
    """SweepConfig should generate correct number of combinations."""
    base = Job(name="sweep", cluster="expanse", repo_path="/repo", command="")
    sweep = SweepConfig(
        base=base,
        command_template="python train.py adapter={adapter} model={model}",
        parameters={"adapter": ["lora", "ia3", "dora"], "model": ["labram", "eegpt"]},
    )
    assert sweep.n_jobs == 6
    combos = sweep.combinations
    assert {"adapter": "lora", "model": "labram"} in combos
    assert {"adapter": "dora", "model": "eegpt"} in combos


def test_sweep_script_generation():
    """Sweep script should include array spec and parameter parsing."""
    base = Job(
        name="adapter_sweep",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
        command="",
        venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
    )
    sweep = SweepConfig(
        base=base,
        command_template="python scripts/train.py adapter={adapter} model={model} data={data}",
        parameters={"adapter": ["lora", "ia3"], "model": ["labram"], "data": ["bcic2a", "physionet"]},
        max_concurrent=2,
    )
    script = sweep.generate_script()

    assert "#SBATCH --array=0-3%2" in script
    assert "EXPERIMENTS=(" in script
    assert "IFS=',' read -r ADAPTER MODEL DATA" in script
    assert "python scripts/train.py adapter=$ADAPTER model=$MODEL data=$DATA" in script
    assert "cd /expanse/projects/nemar/dtyoung/OpenEEG-Bench" in script
    assert "module purge" in script


def test_no_slurm_for_non_hpc():
    """Submitting a SLURM job to a non-SLURM cluster should raise."""
    from neurolab.jobs.submit import _render_slurm_script
    job = Job(name="test", cluster="jamming", repo_path="/repo", command="echo")
    cfg = get_cluster("jamming")
    with pytest.raises(AssertionError):
        _render_slurm_script(job, cfg)

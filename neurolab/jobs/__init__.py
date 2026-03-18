"""neurolab-jobs: Remote job orchestration for HPC and GPU clusters.

Includes cluster configuration, SSH-based job submission, SLURM
monitoring, log retrieval, and parameter sweeps.

All jobs are submitted via SSH using pre-established ControlMaster
connections. SLURM clusters get batch scripts; non-SLURM environments
get direct execution.

Usage:
    from neurolab.jobs import Job, get_cluster

    # Define a job
    job = Job(
        name="train_labram_lora",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
        branch="main",
        venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        env_vars={"WANDB_MODE": "online"},
    )

    # Preview the generated script
    print(job.submit(dry_run=True))

    # Submit for real (via ssh expanse)
    job_id = job.submit()

    # Monitor and get logs
    from neurolab.jobs import monitor_jobs, get_logs
    statuses = monitor_jobs([job_id], cluster="expanse")
    logs = get_logs(job_id, cluster="expanse", repo_path=job.repo_path)

    # Parameter sweeps
    from neurolab.jobs import SweepConfig
    sweep = SweepConfig(
        base=job,
        command_template="python scripts/train.py adapter={adapter} model={model}",
        parameters={"adapter": ["lora", "ia3"], "model": ["labram", "eegpt"]},
    )
    sweep_id = sweep.submit()
"""

# ── Cluster configuration (merged from neurolab-config) ─────────────
from neurolab.jobs.config import (
    ClusterConfig,
    ClusterPaths,
    SlurmDefaults,
    auto_detect_cluster,
    get_cluster,
    list_clusters,
    register_cluster_profile,
)
from neurolab.jobs.environment import EnvironmentManager

# ── Job submission ───────────────────────────────────────────────────
from neurolab.jobs.submit import Job, ssh_run

# ── Monitoring ───────────────────────────────────────────────────────
from neurolab.jobs.monitor import JobStatus, JobState, monitor_jobs, wait_for_jobs

# ── Logs ─────────────────────────────────────────────────────────────
from neurolab.jobs.logs import get_logs, tail_logs, LogEntry, LogResult

# ── Sweeps ───────────────────────────────────────────────────────────
from neurolab.jobs.sweep import SweepConfig, submit_sweep, generate_sweep_script

__all__ = [
    # Config
    "ClusterConfig",
    "ClusterPaths",
    "SlurmDefaults",
    "auto_detect_cluster",
    "get_cluster",
    "list_clusters",
    "register_cluster_profile",
    "EnvironmentManager",
    # Jobs
    "Job",
    "ssh_run",
    # Monitoring
    "JobStatus",
    "JobState",
    "monitor_jobs",
    "wait_for_jobs",
    # Logs
    "get_logs",
    "tail_logs",
    "LogEntry",
    "LogResult",
    # Sweeps
    "SweepConfig",
    "submit_sweep",
    "generate_sweep_script",
]

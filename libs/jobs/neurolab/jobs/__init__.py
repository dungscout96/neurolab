"""neurolab-jobs: SLURM job orchestration for HPC clusters.

Manages job submission, monitoring, log retrieval, and parameter sweeps
for Expanse and Delta clusters.

Usage:
    from neurolab.jobs import SlurmJob, submit_job, monitor_jobs, get_logs

    # Submit a single job
    job = SlurmJob(
        name="train_labram_lora",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        cluster="expanse",
        time_limit="04:00:00",
        gpus=1,
    )
    job_id = submit_job(job)

    # Monitor running jobs
    statuses = monitor_jobs(job_ids=[job_id], cluster="expanse")

    # Get logs for debugging
    stdout, stderr = get_logs(job_id, cluster="expanse")

    # Parameter sweeps
    from neurolab.jobs import SweepConfig, submit_sweep

    sweep = SweepConfig(
        name="adapter_sweep",
        command_template="python scripts/train.py adapter={adapter} model={model}",
        parameters={"adapter": ["lora", "ia3", "dora"], "model": ["labram", "eegpt"]},
        max_concurrent=4,
    )
    job_ids = submit_sweep(sweep, cluster="expanse")
"""

from neurolab.jobs.submit import SlurmJob, submit_job, submit_script
from neurolab.jobs.monitor import JobStatus, monitor_jobs, wait_for_jobs
from neurolab.jobs.logs import get_logs, tail_logs, LogEntry
from neurolab.jobs.sweep import SweepConfig, submit_sweep, generate_sweep_script

__all__ = [
    "SlurmJob",
    "submit_job",
    "submit_script",
    "JobStatus",
    "monitor_jobs",
    "wait_for_jobs",
    "get_logs",
    "tail_logs",
    "LogEntry",
    "SweepConfig",
    "submit_sweep",
    "generate_sweep_script",
]

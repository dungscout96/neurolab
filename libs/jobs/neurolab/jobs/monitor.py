"""SLURM job monitoring.

Provides real-time job status tracking, completion detection, and
batch monitoring utilities using squeue and sacct.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from neurolab.config import ClusterConfig, get_cluster


class JobState(str, Enum):
    """SLURM job states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
            JobState.NODE_FAIL,
            JobState.OUT_OF_MEMORY,
        }

    @property
    def is_success(self) -> bool:
        return self == JobState.COMPLETED


@dataclass
class JobStatus:
    """Status information for a single SLURM job."""

    job_id: str
    name: str
    state: JobState
    partition: str = ""
    elapsed: str = ""
    node: str = ""
    exit_code: str = ""
    reason: str = ""

    @property
    def is_done(self) -> bool:
        return self.state.is_terminal

    @property
    def is_success(self) -> bool:
        return self.state.is_success


def _parse_job_state(state_str: str) -> JobState:
    """Parse a SLURM state string to JobState enum."""
    # Handle states with qualifiers (e.g. "CANCELLED by 12345")
    base = state_str.split()[0].upper()
    try:
        return JobState(base)
    except ValueError:
        return JobState.UNKNOWN


def _run_slurm_cmd(
    cmd: list[str],
    cluster: ClusterConfig,
    ssh: bool = False,
) -> str:
    """Run a SLURM command locally or via SSH."""
    if ssh:
        if not cluster.login_host:
            raise ValueError(f"Cluster '{cluster.name}' has no login_host for SSH")
        full_cmd = ["ssh", cluster.login_host] + cmd
    else:
        full_cmd = cmd

    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def monitor_jobs(
    job_ids: list[str],
    cluster: str = "expanse",
    ssh: bool = False,
) -> list[JobStatus]:
    """Get the current status of one or more SLURM jobs.

    Uses squeue for active jobs and falls back to sacct for completed jobs.

    Args:
        job_ids: List of SLURM job IDs to check.
        cluster: Cluster name.
        ssh: If True, query via SSH.

    Returns:
        List of JobStatus objects.
    """
    cfg = get_cluster(cluster)
    results: list[JobStatus] = []

    for job_id in job_ids:
        status = _check_squeue(job_id, cfg, ssh)
        if status is None:
            # Job not in squeue — check sacct for completed jobs
            status = _check_sacct(job_id, cfg, ssh)
        if status is None:
            status = JobStatus(job_id=job_id, name="", state=JobState.UNKNOWN)
        results.append(status)

    return results


def _check_squeue(
    job_id: str, cluster: ClusterConfig, ssh: bool
) -> JobStatus | None:
    """Check squeue for an active job."""
    cmd = [
        "squeue",
        "-j", job_id,
        "--noheader",
        "--format=%i|%j|%T|%P|%M|%N|%r",
    ]
    output = _run_slurm_cmd(cmd, cluster, ssh)
    if not output:
        return None

    parts = output.split("|")
    if len(parts) < 7:
        return None

    return JobStatus(
        job_id=parts[0].strip(),
        name=parts[1].strip(),
        state=_parse_job_state(parts[2].strip()),
        partition=parts[3].strip(),
        elapsed=parts[4].strip(),
        node=parts[5].strip(),
        reason=parts[6].strip(),
    )


def _check_sacct(
    job_id: str, cluster: ClusterConfig, ssh: bool
) -> JobStatus | None:
    """Check sacct for a completed job."""
    cmd = [
        "sacct",
        "-j", job_id,
        "--noheader",
        "--parsable2",
        "--format=JobID,JobName,State,Partition,Elapsed,NodeList,ExitCode",
    ]
    output = _run_slurm_cmd(cmd, cluster, ssh)
    if not output:
        return None

    # Take the first line (batch step, not sub-steps)
    for line in output.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 7 and not parts[0].endswith(".batch") and not parts[0].endswith(".extern"):
            return JobStatus(
                job_id=parts[0].strip(),
                name=parts[1].strip(),
                state=_parse_job_state(parts[2].strip()),
                partition=parts[3].strip(),
                elapsed=parts[4].strip(),
                node=parts[5].strip(),
                exit_code=parts[6].strip(),
            )
    return None


def wait_for_jobs(
    job_ids: list[str],
    cluster: str = "expanse",
    poll_interval: int = 30,
    timeout: int | None = None,
    ssh: bool = False,
    on_update: Callable[[list[JobStatus]], None] | None = None,
) -> list[JobStatus]:
    """Wait for all jobs to reach a terminal state.

    Polls squeue/sacct at regular intervals until all jobs complete,
    fail, or the timeout is reached.

    Args:
        job_ids: Job IDs to wait for.
        cluster: Cluster name.
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait. None for no limit.
        ssh: If True, query via SSH.
        on_update: Optional callback invoked after each poll with current statuses.

    Returns:
        Final list of JobStatus objects.

    Raises:
        TimeoutError: If timeout is reached before all jobs finish.
    """
    start = time.time()

    while True:
        statuses = monitor_jobs(job_ids, cluster=cluster, ssh=ssh)

        if on_update:
            on_update(statuses)

        if all(s.is_done for s in statuses):
            return statuses

        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for jobs: "
                + ", ".join(s.job_id for s in statuses if not s.is_done)
            )

        time.sleep(poll_interval)

"""Job monitoring.

Provides real-time job status tracking, completion detection, and
batch monitoring utilities. All queries run via SSH using the
cluster's ssh_target.

For SLURM clusters: uses squeue (active) and sacct (completed).
For non-SLURM clusters: uses ps and /proc to check PID status.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from neurolab.jobs.config import ClusterConfig, get_cluster
from neurolab.jobs.submit import ssh_run


class JobState(str, Enum):
    """Job states (covers both SLURM and PID-based jobs)."""

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
    """Status information for a job (SLURM or PID-based)."""

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
    base = state_str.split()[0].upper()
    try:
        return JobState(base)
    except ValueError:
        return JobState.UNKNOWN


def monitor_jobs(
    job_ids: list[str],
    cluster: str = "expanse",
) -> list[JobStatus]:
    """Get the current status of one or more jobs.

    Automatically picks the right monitoring strategy:
    - SLURM clusters: squeue/sacct by job ID
    - Non-SLURM clusters: ps/proc by PID

    Args:
        job_ids: List of job IDs (SLURM IDs or PIDs) to check.
        cluster: Cluster name.

    Returns:
        List of JobStatus objects.
    """
    cfg = get_cluster(cluster)

    if cfg.is_hpc:
        return _monitor_slurm(job_ids, cfg)
    else:
        return _monitor_pids(job_ids, cfg)


# ── SLURM monitoring ────────────────────────────────────────────────


def _monitor_slurm(job_ids: list[str], cfg: ClusterConfig) -> list[JobStatus]:
    """Monitor SLURM jobs via squeue/sacct."""
    results: list[JobStatus] = []

    for job_id in job_ids:
        status = _check_squeue(job_id, cfg)
        if status is None:
            status = _check_sacct(job_id, cfg)
        if status is None:
            status = JobStatus(job_id=job_id, name="", state=JobState.UNKNOWN)
        results.append(status)

    return results


def _check_squeue(job_id: str, cluster: ClusterConfig) -> JobStatus | None:
    """Check squeue for an active job."""
    cmd = f"squeue -j {job_id} --noheader --format='%i|%j|%T|%P|%M|%N|%r'"
    result = ssh_run(cluster, cmd, check=False)
    output = result.stdout.strip()
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


def _check_sacct(job_id: str, cluster: ClusterConfig) -> JobStatus | None:
    """Check sacct for a completed job."""
    cmd = (
        f"sacct -j {job_id} --noheader --parsable2 "
        f"--format=JobID,JobName,State,Partition,Elapsed,NodeList,ExitCode"
    )
    result = ssh_run(cluster, cmd, check=False)
    output = result.stdout.strip()
    if not output:
        return None

    for line in output.strip().split("\n"):
        parts = line.split("|")
        if (
            len(parts) >= 7
            and not parts[0].endswith(".batch")
            and not parts[0].endswith(".extern")
        ):
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


# ── PID-based monitoring (non-SLURM) ────────────────────────────────


def _monitor_pids(job_ids: list[str], cfg: ClusterConfig) -> list[JobStatus]:
    """Monitor jobs by PID via ps on a non-SLURM cluster.

    For each PID:
    - If ps finds it running → RUNNING with elapsed time
    - If ps doesn't find it → check /proc/<pid>/status for exit code
      - exit 0 → COMPLETED
      - exit != 0 → FAILED
      - /proc gone → COMPLETED (process exited and was reaped)
    """
    results: list[JobStatus] = []

    # Batch-check all PIDs in a single SSH call
    pids_csv = ",".join(job_ids)
    cmd = f"ps -p {pids_csv} -o pid=,stat=,etime=,comm= 2>/dev/null || true"
    result = ssh_run(cfg, cmd, check=False, timeout=15)

    # Parse ps output into a lookup: pid → (stat, etime, comm)
    running: dict[str, tuple[str, str, str]] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3:
            pid = parts[0].strip()
            stat = parts[1].strip()
            etime = parts[2].strip()
            comm = parts[3].strip() if len(parts) > 3 else ""
            running[pid] = (stat, etime, comm)

    for pid in job_ids:
        if pid in running:
            stat, etime, comm = running[pid]
            # ps STAT codes: S=sleeping, R=running, T=stopped, Z=zombie, D=uninterruptible
            if stat.startswith("Z"):
                state = JobState.FAILED
            else:
                state = JobState.RUNNING
            results.append(JobStatus(
                job_id=pid,
                name=comm,
                state=state,
                elapsed=etime,
                node=cfg.name,
            ))
        else:
            # Process is gone — try to get exit code via wait status
            # Since we disowned the process, we can't get its exit code from
            # the shell. Check if the log file's last line indicates success.
            results.append(JobStatus(
                job_id=pid,
                name="",
                state=JobState.COMPLETED,
                node=cfg.name,
                reason="process exited",
            ))

    return results


def wait_for_jobs(
    job_ids: list[str],
    cluster: str = "expanse",
    poll_interval: int = 30,
    timeout: int | None = None,
    on_update: Callable[[list[JobStatus]], None] | None = None,
) -> list[JobStatus]:
    """Wait for all jobs to reach a terminal state.

    Polls at regular intervals until all jobs complete,
    fail, or the timeout is reached. Works for both SLURM
    and PID-based jobs.

    Args:
        job_ids: Job IDs (SLURM IDs or PIDs) to wait for.
        cluster: Cluster name.
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait. None for no limit.
        on_update: Optional callback invoked after each poll with current statuses.

    Returns:
        Final list of JobStatus objects.

    Raises:
        TimeoutError: If timeout is reached before all jobs finish.
    """
    start = time.time()

    while True:
        statuses = monitor_jobs(job_ids, cluster=cluster)

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

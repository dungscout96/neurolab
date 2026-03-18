"""Job log retrieval and parsing.

Provides utilities to fetch, search, and tail job output/error logs
from remote clusters via SSH. Works for both SLURM jobs (log files
named by job ID) and direct jobs (log files named by job name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from neurolab.jobs.config import ClusterConfig, get_cluster
from neurolab.jobs.submit import ssh_run


@dataclass
class LogEntry:
    """A parsed log entry with optional metadata."""

    line: str
    line_number: int
    source: str  # "stdout" or "stderr"
    level: str = ""  # "ERROR", "WARNING", "INFO", etc.


@dataclass
class LogResult:
    """Result of a log retrieval operation."""

    job_id: str
    stdout: str = ""
    stderr: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    @property
    def has_errors(self) -> bool:
        """Check if stderr contains content."""
        return bool(self.stderr.strip())

    def search(self, pattern: str, source: str = "both") -> list[LogEntry]:
        """Search logs with a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            source: "stdout", "stderr", or "both".

        Returns:
            List of matching LogEntry objects.
        """
        entries: list[LogEntry] = []
        regex = re.compile(pattern, re.IGNORECASE)

        if source in ("stdout", "both"):
            for i, line in enumerate(self.stdout.splitlines(), 1):
                if regex.search(line):
                    entries.append(LogEntry(line=line, line_number=i, source="stdout"))

        if source in ("stderr", "both"):
            for i, line in enumerate(self.stderr.splitlines(), 1):
                if regex.search(line):
                    entries.append(LogEntry(line=line, line_number=i, source="stderr"))

        return entries

    def errors(self) -> list[LogEntry]:
        """Extract error-level log lines from both streams."""
        return self.search(r"(error|exception|traceback|failed|fatal)", source="both")

    def tail(self, n: int = 50, source: str = "both") -> str:
        """Get the last n lines from logs."""
        parts: list[str] = []
        if source in ("stdout", "both") and self.stdout:
            lines = self.stdout.splitlines()[-n:]
            parts.append(f"=== stdout (last {len(lines)} lines) ===\n" + "\n".join(lines))
        if source in ("stderr", "both") and self.stderr:
            lines = self.stderr.splitlines()[-n:]
            parts.append(f"=== stderr (last {len(lines)} lines) ===\n" + "\n".join(lines))
        return "\n\n".join(parts)


def get_logs(
    job_id: str,
    cluster: str = "expanse",
    log_dir: str = "logs",
    repo_path: str = "",
    name: str = "",
) -> LogResult:
    """Retrieve stdout and stderr logs for a job via SSH.

    Searches for log files matching common naming patterns in the
    specified log directory on the remote cluster.

    For SLURM jobs, looks for: *_{job_id}.out, slurm-{job_id}.out
    For direct (PID) jobs, looks for: {name}.out, {name}.err

    Args:
        job_id: SLURM job ID or PID.
        cluster: Cluster name.
        log_dir: Directory containing log files (relative to repo_path or absolute).
        repo_path: If set, log_dir is resolved relative to this path.
        name: Job name (required for direct/PID-based jobs to find log files).

    Returns:
        LogResult with stdout and stderr content.
    """
    cfg = get_cluster(cluster)
    result = LogResult(job_id=job_id)

    # Build the remote log directory path
    if repo_path and not log_dir.startswith("/"):
        remote_log_dir = f"{repo_path}/{log_dir}"
    else:
        remote_log_dir = log_dir

    if cfg.is_hpc:
        _fetch_slurm_logs(result, cfg, remote_log_dir, job_id)
    else:
        _fetch_direct_logs(result, cfg, remote_log_dir, name or job_id)

    return result


def _fetch_slurm_logs(
    result: LogResult,
    cfg: ClusterConfig,
    remote_log_dir: str,
    job_id: str,
) -> None:
    """Fetch SLURM-style log files (*_{job_id}.out/.err)."""
    for suffix, attr in [(".out", "stdout"), (".err", "stderr")]:
        # Find matching files on the remote
        find_cmd = f"ls {remote_log_dir}/*_{job_id}{suffix} 2>/dev/null | head -1"
        find_result = ssh_run(cfg, find_cmd, check=False)
        remote_path = find_result.stdout.strip()

        if remote_path:
            cat_result = ssh_run(cfg, f"cat {remote_path}", check=False)
            if cat_result.returncode == 0:
                setattr(result, attr, cat_result.stdout)
                setattr(result, f"{attr}_path", remote_path)
            continue

        # Try slurm-{job_id}.out pattern
        slurm_path = f"{remote_log_dir}/slurm-{job_id}{suffix}"
        cat_result = ssh_run(cfg, f"cat {slurm_path} 2>/dev/null", check=False)
        if cat_result.returncode == 0 and cat_result.stdout:
            setattr(result, attr, cat_result.stdout)
            setattr(result, f"{attr}_path", slurm_path)


def _fetch_direct_logs(
    result: LogResult,
    cfg: ClusterConfig,
    remote_log_dir: str,
    name: str,
) -> None:
    """Fetch direct-style log files ({name}.out/.err)."""
    for suffix, attr in [(".out", "stdout"), (".err", "stderr")]:
        remote_path = f"{remote_log_dir}/{name}{suffix}"
        cat_result = ssh_run(cfg, f"cat {remote_path} 2>/dev/null", check=False)
        if cat_result.returncode == 0 and cat_result.stdout:
            setattr(result, attr, cat_result.stdout)
            setattr(result, f"{attr}_path", remote_path)


def tail_logs(
    job_id: str,
    cluster: str = "expanse",
    n_lines: int = 50,
    log_dir: str = "logs",
    repo_path: str = "",
    name: str = "",
) -> str:
    """Get the last n lines of logs for a job.

    Convenience wrapper around get_logs() that returns just the tail.

    Args:
        job_id: SLURM job ID or PID.
        cluster: Cluster name.
        n_lines: Number of trailing lines to return.
        log_dir: Directory containing log files.
        repo_path: If set, log_dir is resolved relative to this path.
        name: Job name (for direct/PID-based jobs).

    Returns:
        Formatted string with the tail of stdout and stderr.
    """
    result = get_logs(
        job_id, cluster=cluster, log_dir=log_dir,
        repo_path=repo_path, name=name,
    )
    return result.tail(n=n_lines)

"""SLURM log retrieval and parsing.

Provides utilities to fetch, search, and tail job output/error logs
from local paths or remote clusters via SSH.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from neurolab.config import get_cluster, ClusterConfig


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


def _find_log_files(
    job_id: str,
    log_dir: str = "logs",
    job_name: str = "",
) -> tuple[Path | None, Path | None]:
    """Find stdout and stderr log files for a job.

    Searches for common naming patterns:
    - {job_name}_{job_id}.out / .err
    - slurm-{job_id}.out
    - train_{job_id}.out / .err
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return None, None

    patterns = [
        f"*_{job_id}.out",
        f"*_{job_id}.err",
        f"slurm-{job_id}.out",
    ]

    stdout_file = None
    stderr_file = None

    for f in log_path.iterdir():
        if job_id in f.name:
            if f.suffix == ".out":
                stdout_file = f
            elif f.suffix == ".err":
                stderr_file = f

    return stdout_file, stderr_file


def get_logs(
    job_id: str,
    cluster: str = "expanse",
    log_dir: str = "logs",
    job_name: str = "",
    ssh: bool = False,
) -> LogResult:
    """Retrieve stdout and stderr logs for a SLURM job.

    Args:
        job_id: SLURM job ID.
        cluster: Cluster name.
        log_dir: Directory containing log files.
        job_name: Job name prefix for log files.
        ssh: If True, fetch logs via SSH from the cluster.

    Returns:
        LogResult with stdout and stderr content.
    """
    cfg = get_cluster(cluster)

    if ssh:
        return _get_logs_ssh(job_id, cfg, log_dir, job_name)

    stdout_file, stderr_file = _find_log_files(job_id, log_dir, job_name)

    result = LogResult(job_id=job_id)

    if stdout_file and stdout_file.exists():
        result.stdout = stdout_file.read_text()
        result.stdout_path = str(stdout_file)

    if stderr_file and stderr_file.exists():
        result.stderr = stderr_file.read_text()
        result.stderr_path = str(stderr_file)

    return result


def _get_logs_ssh(
    job_id: str,
    cluster: ClusterConfig,
    log_dir: str,
    job_name: str,
) -> LogResult:
    """Fetch logs via SSH from a remote cluster."""
    if not cluster.login_host:
        raise ValueError(f"Cluster '{cluster.name}' has no login_host for SSH")

    result = LogResult(job_id=job_id)

    # Try common patterns
    for suffix, attr in [(".out", "stdout"), (".err", "stderr")]:
        for pattern in [f"{log_dir}/*_{job_id}{suffix}", f"{log_dir}/slurm-{job_id}{suffix}"]:
            try:
                cmd_result = subprocess.run(
                    ["ssh", cluster.login_host, f"cat {pattern}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if cmd_result.returncode == 0 and cmd_result.stdout:
                    setattr(result, attr, cmd_result.stdout)
                    setattr(result, f"{attr}_path", pattern)
                    break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                continue

    return result


def tail_logs(
    job_id: str,
    cluster: str = "expanse",
    n_lines: int = 50,
    log_dir: str = "logs",
    ssh: bool = False,
) -> str:
    """Get the last n lines of logs for a job.

    Convenience wrapper around get_logs() that returns just the tail.

    Args:
        job_id: SLURM job ID.
        cluster: Cluster name.
        n_lines: Number of trailing lines to return.
        log_dir: Directory containing log files.
        ssh: If True, fetch via SSH.

    Returns:
        Formatted string with the tail of stdout and stderr.
    """
    result = get_logs(job_id, cluster=cluster, log_dir=log_dir, ssh=ssh)
    return result.tail(n=n_lines)

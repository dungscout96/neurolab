"""CLI entry points for neurolab-jobs.

Provides command-line tools for job submission, status checking, and
log retrieval without needing to write Python scripts.

Usage:
    neurolab-submit --name train_run --command "python train.py" --cluster expanse
    neurolab-status 12345678 --cluster expanse
    neurolab-logs 12345678 --cluster expanse --tail 100
"""

from __future__ import annotations

import argparse
import sys

from neurolab.jobs.submit import SlurmJob, submit_job
from neurolab.jobs.monitor import monitor_jobs, JobState
from neurolab.jobs.logs import get_logs, tail_logs


def main_submit() -> None:
    """CLI for submitting SLURM jobs."""
    parser = argparse.ArgumentParser(
        description="Submit a SLURM job via neurolab",
        prog="neurolab-submit",
    )
    parser.add_argument("--name", required=True, help="Job name")
    parser.add_argument("--command", required=True, help="Command to execute")
    parser.add_argument("--cluster", default="expanse", help="Cluster name")
    parser.add_argument("--time", default=None, help="Time limit (HH:MM:SS)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--working-dir", default="", help="Working directory")
    parser.add_argument("--dry-run", action="store_true", help="Generate script only")
    parser.add_argument("--ssh", action="store_true", help="Submit via SSH")

    args = parser.parse_args()

    job = SlurmJob(
        name=args.name,
        command=args.command,
        cluster=args.cluster,
        time_limit=args.time,
        gpus=args.gpus,
        log_dir=args.log_dir,
        working_dir=args.working_dir,
    )

    result = submit_job(job, dry_run=args.dry_run, ssh=args.ssh)

    if args.dry_run:
        print(f"Script written to: {result}")
    else:
        print(f"Submitted job: {result}")


def main_status() -> None:
    """CLI for checking SLURM job status."""
    parser = argparse.ArgumentParser(
        description="Check SLURM job status",
        prog="neurolab-status",
    )
    parser.add_argument("job_ids", nargs="+", help="SLURM job IDs")
    parser.add_argument("--cluster", default="expanse", help="Cluster name")
    parser.add_argument("--ssh", action="store_true", help="Query via SSH")

    args = parser.parse_args()

    statuses = monitor_jobs(args.job_ids, cluster=args.cluster, ssh=args.ssh)

    for s in statuses:
        icon = {"RUNNING": "⏳", "COMPLETED": "✅", "FAILED": "❌", "PENDING": "🕐"}.get(
            s.state.value, "❓"
        )
        parts = [f"{icon} {s.job_id}", s.state.value]
        if s.name:
            parts.append(s.name)
        if s.elapsed:
            parts.append(s.elapsed)
        if s.node:
            parts.append(s.node)
        print(" | ".join(parts))


def main_logs() -> None:
    """CLI for retrieving SLURM job logs."""
    parser = argparse.ArgumentParser(
        description="Retrieve SLURM job logs",
        prog="neurolab-logs",
    )
    parser.add_argument("job_id", help="SLURM job ID")
    parser.add_argument("--cluster", default="expanse", help="Cluster name")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--tail", type=int, default=0, help="Show last N lines")
    parser.add_argument("--errors", action="store_true", help="Show only errors")
    parser.add_argument("--ssh", action="store_true", help="Fetch via SSH")

    args = parser.parse_args()

    if args.tail:
        print(tail_logs(args.job_id, cluster=args.cluster, n_lines=args.tail,
                        log_dir=args.log_dir, ssh=args.ssh))
    else:
        result = get_logs(args.job_id, cluster=args.cluster, log_dir=args.log_dir, ssh=args.ssh)
        if args.errors:
            errors = result.errors()
            for e in errors:
                print(f"[{e.source}:{e.line_number}] {e.line}")
            if not errors:
                print("No errors found.")
        else:
            if result.stdout:
                print("=== STDOUT ===")
                print(result.stdout)
            if result.stderr:
                print("\n=== STDERR ===")
                print(result.stderr)
            if not result.stdout and not result.stderr:
                print(f"No log files found for job {args.job_id}")

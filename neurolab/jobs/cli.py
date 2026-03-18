"""CLI entry points for neurolab-jobs.

Provides command-line tools for job submission, status checking, and
log retrieval.

Usage:
    neurolab-submit --name train_run --cluster expanse \\
        --repo /expanse/projects/nemar/dtyoung/OpenEEG-Bench \\
        --command "python scripts/train.py adapter=lora" \\
        --venv /expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning

    neurolab-status 12345678 --cluster expanse
    neurolab-logs 12345678 --cluster expanse --tail 100
"""

from __future__ import annotations

import argparse
import sys

from neurolab.jobs.submit import Job
from neurolab.jobs.monitor import monitor_jobs
from neurolab.jobs.logs import get_logs, tail_logs


def main_submit() -> None:
    """CLI for submitting jobs."""
    parser = argparse.ArgumentParser(
        description="Submit a job via neurolab (SSH + optional SLURM)",
        prog="neurolab-submit",
    )
    parser.add_argument("--name", required=True, help="Job name")
    parser.add_argument("--cluster", required=True, help="Cluster name (e.g. expanse, delta, jamming)")
    parser.add_argument("--repo", required=True, help="Absolute path to project repo on the remote system")
    parser.add_argument("--command", required=True, help="Entry point command to run")
    parser.add_argument("--venv", default="", help="Path to conda/venv on the remote system")
    parser.add_argument("--branch", default="main", help="Git branch to checkout (default: main)")
    parser.add_argument("--time", default=None, help="Time limit for SLURM (HH:MM:SS)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (SLURM)")
    parser.add_argument("--env", nargs="*", default=[], help="Extra env vars as KEY=VALUE pairs")
    parser.add_argument("--dry-run", action="store_true", help="Print script without submitting")

    args = parser.parse_args()

    # Parse env vars
    env_vars = {}
    for pair in args.env:
        if "=" in pair:
            k, v = pair.split("=", 1)
            env_vars[k] = v

    job = Job(
        name=args.name,
        cluster=args.cluster,
        repo_path=args.repo,
        command=args.command,
        venv=args.venv,
        branch=args.branch,
        time_limit=args.time,
        gpus=args.gpus,
        env_vars=env_vars,
    )

    result = job.submit(dry_run=args.dry_run)

    if args.dry_run:
        print(result)
    else:
        print(f"Submitted: {result}")


def main_status() -> None:
    """CLI for checking SLURM job status."""
    parser = argparse.ArgumentParser(
        description="Check SLURM job status via SSH",
        prog="neurolab-status",
    )
    parser.add_argument("job_ids", nargs="+", help="SLURM job IDs")
    parser.add_argument("--cluster", default="expanse", help="Cluster name")

    args = parser.parse_args()
    statuses = monitor_jobs(args.job_ids, cluster=args.cluster)

    for s in statuses:
        icon = {"RUNNING": "~", "COMPLETED": "+", "FAILED": "X", "PENDING": "?"}.get(
            s.state.value, "?"
        )
        parts = [f"[{icon}] {s.job_id}", s.state.value]
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
        description="Retrieve SLURM job logs via SSH",
        prog="neurolab-logs",
    )
    parser.add_argument("job_id", help="SLURM job ID")
    parser.add_argument("--cluster", default="expanse", help="Cluster name")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--repo", default="", help="Repo path (log-dir resolved relative to this)")
    parser.add_argument("--tail", type=int, default=0, help="Show last N lines")
    parser.add_argument("--errors", action="store_true", help="Show only errors")

    args = parser.parse_args()

    if args.tail:
        print(tail_logs(
            args.job_id, cluster=args.cluster, n_lines=args.tail,
            log_dir=args.log_dir, repo_path=args.repo,
        ))
    else:
        result = get_logs(
            args.job_id, cluster=args.cluster,
            log_dir=args.log_dir, repo_path=args.repo,
        )
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

"""SLURM job submission.

Provides a high-level interface for submitting jobs to SLURM clusters,
handling script generation, resource configuration, and sbatch invocation
(locally or via SSH for remote submission).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

from jinja2 import Environment, FileSystemLoader, BaseLoader

from neurolab.config import ClusterConfig, get_cluster


# ── Jinja2 templates ────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_INLINE_TEMPLATE = dedent("""\
    #!/usr/bin/env bash
    #SBATCH --job-name={{ job.name }}
    #SBATCH --partition={{ slurm.partition }}
    #SBATCH --account={{ slurm.account }}
    #SBATCH --nodes={{ slurm.nodes }}
    #SBATCH --ntasks-per-node={{ slurm.tasks_per_node }}
    #SBATCH --cpus-per-task={{ slurm.cpus_per_task }}
    #SBATCH --gpus={{ job.gpus }}
    #SBATCH --mem={{ slurm.mem_gb }}G
    #SBATCH --time={{ job.time_limit }}
    #SBATCH --output={{ job.log_dir }}/{{ job.name }}_%j.out
    #SBATCH --error={{ job.log_dir }}/{{ job.name }}_%j.err
    {% if job.array_spec %}#SBATCH --array={{ job.array_spec }}{% endif %}
    {% if slurm.qos %}#SBATCH --qos={{ slurm.qos }}{% endif %}
    {% if job.dependency %}#SBATCH --dependency={{ job.dependency }}{% endif %}
    {% if job.export_vars %}#SBATCH --export={{ job.export_vars }}{% endif %}

    # ── Environment setup ──
    {{ env_setup }}

    # ── Working directory ──
    {% if job.working_dir %}cd {{ job.working_dir }}{% endif %}

    # ── Command ──
    {{ job.command }}
""")


@dataclass
class SlurmJob:
    """Specification for a single SLURM job.

    Attributes:
        name: Job name (used for --job-name and log file naming).
        command: The shell command(s) to execute.
        cluster: Cluster name to submit to (resolved via neurolab-config).
        time_limit: Wall clock limit (overrides cluster default if set).
        gpus: Number of GPUs to request.
        log_dir: Directory for stdout/stderr logs.
        working_dir: Working directory for the job.
        array_spec: SLURM array specification (e.g. "0-191%4").
        dependency: Job dependency string (e.g. "afterok:12345").
        extra_sbatch: Additional #SBATCH lines to include.
        env_overrides: Environment variables that override cluster defaults.
    """

    name: str
    command: str
    cluster: str = "expanse"
    time_limit: str | None = None
    gpus: int = 1
    log_dir: str = "logs"
    working_dir: str = ""
    array_spec: str = ""
    dependency: str = ""
    extra_sbatch: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    export_vars: str = ""

    def resolve_cluster(self) -> ClusterConfig:
        """Resolve the cluster configuration."""
        return get_cluster(self.cluster)


def _render_job_script(job: SlurmJob) -> str:
    """Render a SLURM batch script from a job specification."""
    cluster = job.resolve_cluster()
    if cluster.slurm is None:
        raise ValueError(f"Cluster '{cluster.name}' has no SLURM configuration")

    slurm = cluster.slurm

    # Apply env overrides
    merged_env = {**cluster.env_vars, **job.env_overrides}

    # Build export string if not explicitly set
    if not job.export_vars and merged_env:
        job.export_vars = ",".join(f"{k}={v}" for k, v in sorted(merged_env.items()))

    # Override time limit
    effective_time = job.time_limit or slurm.time_limit

    env_setup = cluster.render_env_setup()

    # Render with Jinja2
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    template = env.from_string(_INLINE_TEMPLATE)

    # Create a copy of job with resolved time limit for template rendering
    return template.render(
        job=SlurmJob(
            name=job.name,
            command=job.command,
            cluster=job.cluster,
            time_limit=effective_time,
            gpus=job.gpus,
            log_dir=job.log_dir,
            working_dir=job.working_dir,
            array_spec=job.array_spec,
            dependency=job.dependency,
            extra_sbatch=job.extra_sbatch,
            env_overrides=job.env_overrides,
            export_vars=job.export_vars,
        ),
        slurm=slurm,
        env_setup=env_setup,
    )


def submit_job(
    job: SlurmJob,
    dry_run: bool = False,
    script_dir: str | Path | None = None,
    ssh: bool = False,
) -> str:
    """Submit a SLURM job.

    Args:
        job: Job specification.
        dry_run: If True, write the script but don't submit. Returns the script path.
        script_dir: Directory to write the generated script. Defaults to a temp dir.
        ssh: If True, submit via SSH to the cluster's login node.

    Returns:
        The SLURM job ID (e.g. "12345678"), or the script path if dry_run.
    """
    script_content = _render_job_script(job)

    # Write script to file
    if script_dir:
        out_dir = Path(script_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = out_dir / f"{job.name}.slurm"
    else:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".slurm", prefix=f"{job.name}_", delete=False
        )
        script_path = Path(tmp.name)
        tmp.close()

    script_path.write_text(script_content)

    if dry_run:
        return str(script_path)

    # Ensure log directory exists
    cluster = job.resolve_cluster()
    log_path = Path(job.log_dir)
    if not ssh:
        log_path.mkdir(parents=True, exist_ok=True)

    # Submit
    if ssh:
        return _submit_via_ssh(cluster, script_path)
    else:
        return _submit_local(script_path)


def _submit_local(script_path: Path) -> str:
    """Submit a script using local sbatch."""
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    # sbatch output: "Submitted batch job 12345678"
    return result.stdout.strip().split()[-1]


def _submit_via_ssh(cluster: ClusterConfig, script_path: Path) -> str:
    """Submit a script via SSH to the cluster login node."""
    if not cluster.login_host:
        raise ValueError(f"Cluster '{cluster.name}' has no login_host configured for SSH")

    # Copy script to cluster
    scp_result = subprocess.run(
        ["scp", str(script_path), f"{cluster.login_host}:/tmp/{script_path.name}"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Submit remotely
    result = subprocess.run(
        ["ssh", cluster.login_host, f"sbatch /tmp/{script_path.name}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split()[-1]


def submit_script(
    script_path: str | Path,
    cluster: str = "expanse",
    ssh: bool = False,
) -> str:
    """Submit an existing SLURM script directly.

    Use this for hand-written scripts or scripts generated externally.

    Args:
        script_path: Path to the .slurm script.
        cluster: Cluster name (used only for SSH submission).
        ssh: If True, submit via SSH.

    Returns:
        The SLURM job ID.
    """
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    if ssh:
        cfg = get_cluster(cluster)
        return _submit_via_ssh(cfg, path)
    else:
        return _submit_local(path)

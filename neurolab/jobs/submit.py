"""Job submission via SSH.

All jobs are submitted through SSH to the target cluster using
pre-established ControlMaster connections (required for 2FA clusters).

For SLURM clusters (Expanse, Delta): generates a batch script and
submits it via `ssh <alias> sbatch` (piped through stdin).

For non-SLURM environments (jamming): runs the command directly
via `ssh <alias> bash -ls`.

SSH aliases are defined in ~/.ssh/config and referenced by the
cluster profile's `ssh_alias` field:

    Host expanse
        HostName login.expanse.sdsc.edu
        User dtyoung
        ControlMaster auto
        ControlPath ~/.ssh/sockets/%r@%h-%p
        ControlPersist 12h
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from neurolab.jobs.config import ClusterConfig, get_cluster


@dataclass
class Job:
    """A job to run on a remote cluster.

    Encapsulates everything needed to execute an experiment:
    which cluster, which repo checkout, which environment, and
    what command to run.

    Usage:
        job = Job(
            name="train_labram_lora",
            cluster="expanse",
            repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
            venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
            command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        )
        result = job.submit()              # submits via ssh expanse
        script = job.submit(dry_run=True)  # returns the script without submitting
    """

    name: str
    """Job name (used for SLURM --job-name and log file naming)."""

    cluster: str
    """Cluster name — must match a registered profile (e.g. 'expanse', 'delta', 'jamming')."""

    repo_path: str
    """Absolute path to the project repo on the remote system."""

    command: str
    """The entry point command to run (e.g. 'python experiments/main.py --lr 1e-4')."""

    venv: str = ""
    """Path to a virtual/conda environment on the remote system.
    If set, the job will `source activate <venv>` before running.
    If empty, uses the cluster profile's default conda_env.
    Set to '__none__' to explicitly skip activation."""

    branch: str = "main"
    """Git branch to checkout before running. Set to '' to skip branch management."""

    env_vars: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to export for this job.
    Merged with (and overrides) the cluster profile's env_vars."""

    # ── SLURM-specific (ignored for non-SLURM clusters) ─────────────
    time_limit: str | None = None
    """Wall clock limit. Defaults to the cluster profile's default."""

    gpus: int = 1
    """Number of GPUs to request."""

    partition: str | None = None
    """SLURM partition override. Defaults to the cluster profile's default."""

    cpus_per_task: int | None = None
    """CPUs per task override. Defaults to the cluster profile's default."""

    mem_gb: int | None = None
    """Memory in GB override. Defaults to the cluster profile's default."""

    log_dir: str = "logs"
    """Directory for stdout/stderr logs (relative to repo_path)."""

    array_spec: str = ""
    """SLURM array specification (e.g. '0-191%4')."""

    dependency: str = ""
    """SLURM dependency string (e.g. 'afterok:12345')."""

    def resolve_cluster(self) -> ClusterConfig:
        """Resolve the cluster configuration."""
        return get_cluster(self.cluster)

    def submit(self, dry_run: bool = False) -> str:
        """Submit this job to the target cluster via SSH.

        For SLURM clusters: generates a batch script, pipes it to
        `ssh <alias> sbatch` via stdin.

        For non-SLURM clusters: builds a shell command sequence and
        runs it via `ssh <alias> bash -ls`.

        Args:
            dry_run: If True, return the generated script without submitting.

        Returns:
            SLURM job ID (e.g. '12345678') for SLURM clusters,
            or stdout from direct execution, or the script content if dry_run.
        """
        cluster = self.resolve_cluster()

        if cluster.is_hpc:
            return _submit_slurm(self, cluster, dry_run=dry_run)
        else:
            return _submit_direct(self, cluster, dry_run=dry_run)


# ── SSH transport ────────────────────────────────────────────────────


def ssh_run(
    cluster: str | ClusterConfig,
    command: str,
    stdin: str | None = None,
    check: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run a command on a remote cluster via SSH.

    Uses the cluster's ssh_alias (matching ~/.ssh/config Host entry)
    which leverages ControlMaster for pre-authenticated connections.

    This is also useful for ad-hoc commands like checking disk usage,
    listing files, or running quick diagnostics.

    Args:
        cluster: Cluster name or ClusterConfig object.
        command: Remote command to execute.
        stdin: Optional string to pipe to stdin.
        check: If True, raise on non-zero exit code.
        timeout: Timeout in seconds.

    Returns:
        CompletedProcess result.

    Raises:
        ValueError: If the cluster has no SSH target configured.
        subprocess.CalledProcessError: If the command fails and check=True.
    """
    if isinstance(cluster, str):
        cluster = get_cluster(cluster)

    target = cluster.ssh_target
    if not target:
        raise ValueError(
            f"Cluster '{cluster.name}' has no ssh_alias or login_host configured. "
            f"Cannot run remote commands."
        )

    cmd = ["ssh", target, command]
    return subprocess.run(
        cmd,
        input=stdin,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


# ── Script rendering ────────────────────────────────────────────────


def _build_preamble(job: Job, cluster: ClusterConfig) -> str:
    """Build the shell preamble that runs before the job command.

    Order:
    1. cd into repo
    2. checkout branch + pull
    3. module loads (HPC clusters)
    4. activate virtual environment
    5. export environment variables
    """
    lines: list[str] = []

    # 1. cd into repo
    lines.append(f"cd {job.repo_path}")

    # 2. Git branch checkout
    if job.branch:
        lines.append(f"git checkout {job.branch}")
        lines.append("git pull --ff-only 2>/dev/null || true")

    # 3. Module loads (SLURM clusters)
    if cluster.modules:
        lines.append("module purge")
        for mod in cluster.modules:
            lines.append(f"module load {mod}")

    # 4. Virtual environment activation
    venv = job.venv if job.venv else cluster.conda_env
    if venv and venv != "__none__":
        lines.append(f"source activate {venv}")

    # 5. Environment variables: cluster defaults + job overrides
    merged_env = {**cluster.env_vars, **job.env_vars}
    for key, value in sorted(merged_env.items()):
        lines.append(f'export {key}="{value}"')

    return "\n".join(lines)


def _render_slurm_script(job: Job, cluster: ClusterConfig) -> str:
    """Render a complete SLURM batch script."""
    slurm = cluster.slurm
    assert slurm is not None

    time_limit = job.time_limit or slurm.time_limit
    partition = job.partition or slurm.partition
    cpus = job.cpus_per_task or slurm.cpus_per_task
    mem = job.mem_gb or slurm.mem_gb

    preamble = _build_preamble(job, cluster)

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job.name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks-per-node={slurm.tasks_per_node}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --gpus={job.gpus}",
        f"#SBATCH --mem={mem}G",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={job.log_dir}/{job.name}_%j.out",
        f"#SBATCH --error={job.log_dir}/{job.name}_%j.err",
    ]

    if job.array_spec:
        lines.append(f"#SBATCH --array={job.array_spec}")
    if slurm.qos:
        lines.append(f"#SBATCH --qos={slurm.qos}")
    if job.dependency:
        lines.append(f"#SBATCH --dependency={job.dependency}")

    lines.append("")
    lines.append(preamble)
    lines.append("")
    lines.append(f"mkdir -p {job.log_dir}")
    lines.append("")
    lines.append('echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="')
    lines.append("")
    lines.append(job.command)
    lines.append("")
    lines.append('echo "=== Job $SLURM_JOB_ID finished at $(date) ==="')

    return "\n".join(lines) + "\n"


def _render_direct_script(job: Job, cluster: ClusterConfig) -> str:
    """Render a shell script for direct (non-SLURM) execution.

    The script is wrapped with nohup so it runs in the background
    and survives SSH disconnection. Stdout/stderr go to log files
    under log_dir (relative to repo_path). The script prints the
    background PID to stdout so the caller can capture it.
    """
    preamble = _build_preamble(job, cluster)

    log_out = f"{job.log_dir}/{job.name}.out"
    log_err = f"{job.log_dir}/{job.name}.err"

    lines = [
        "set -euo pipefail",
        preamble,
        "",
        f"mkdir -p {job.log_dir}",
        "",
        f'echo "=== Job {job.name} on $(hostname) at $(date) ===" | tee {log_out}',
        "",
        f"nohup bash -c '{job.command}' >> {log_out} 2> {log_err} &",
        "BGPID=$!",
        "disown $BGPID",
        'echo "$BGPID"',
    ]
    return "\n".join(lines) + "\n"


# ── Submission ───────────────────────────────────────────────────────


def _submit_slurm(job: Job, cluster: ClusterConfig, dry_run: bool = False) -> str:
    """Submit a SLURM job by piping the script to `ssh <alias> sbatch`."""
    script = _render_slurm_script(job, cluster)

    if dry_run:
        return script

    # Ensure log dir exists on remote
    ssh_run(cluster, f"mkdir -p {job.repo_path}/{job.log_dir}", check=False)

    # Pipe script to sbatch via stdin — no need to scp a file
    result = ssh_run(cluster, "sbatch", stdin=script)

    # sbatch output: "Submitted batch job 12345678"
    output = result.stdout.strip()
    if "Submitted" in output:
        return output.split()[-1]
    return output


def _submit_direct(job: Job, cluster: ClusterConfig, dry_run: bool = False) -> str:
    """Run a job in the background via SSH (no SLURM scheduler).

    The job is launched with nohup and disowned so it survives SSH
    disconnection. Returns the remote PID (like sbatch returns a job ID).
    Stdout/stderr are redirected to log files under log_dir.
    """
    script = _render_direct_script(job, cluster)

    if dry_run:
        return script

    # The script backgrounds the job and prints its PID.
    # This returns quickly — the job keeps running on the remote.
    result = ssh_run(cluster, "bash -ls", stdin=script, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Job '{job.name}' failed to launch on {cluster.name} "
            f"(exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # Last line of stdout is the PID
    pid = result.stdout.strip().splitlines()[-1]
    return pid

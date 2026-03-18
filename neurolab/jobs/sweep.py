"""Parameter sweep generation and submission.

Creates SLURM array jobs from combinatorial parameter grids,
following the pattern used in OpenEEG-Bench's sweep scripts.

Sweeps are built on top of Job — you provide a base job configuration
and a parameter grid, and the sweep generates a SLURM array job that
iterates over all combinations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from neurolab.jobs.config import get_cluster
from neurolab.jobs.submit import Job, ssh_run, _build_preamble


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep.

    Generates all combinations from the parameter grid and creates
    a SLURM array job. Built on top of the Job model — inherits
    repo_path, branch, venv, and cluster from a base Job spec.

    Usage:
        sweep = SweepConfig(
            base=Job(
                name="adapter_sweep",
                cluster="expanse",
                repo_path="/expanse/projects/nemar/dtyoung/OpenEEG-Bench",
                venv="/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning",
                command="",  # overridden by command_template
            ),
            command_template="python scripts/train.py adapter={adapter} model={model}",
            parameters={"adapter": ["lora", "ia3", "dora"], "model": ["labram", "eegpt"]},
            max_concurrent=4,
        )
        script = sweep.submit(dry_run=True)
    """

    base: Job
    """Base job configuration (cluster, repo_path, branch, venv, etc.)."""

    command_template: str
    """Command with {param_name} placeholders (e.g. 'python train.py adapter={adapter}')."""

    parameters: dict[str, list[str]]
    """Parameter grid: each key maps to a list of values to sweep over."""

    max_concurrent: int = 4
    """Maximum simultaneous array tasks (%N in --array)."""

    @property
    def combinations(self) -> list[dict[str, str]]:
        """Generate all parameter combinations."""
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @property
    def n_jobs(self) -> int:
        return len(self.combinations)

    def generate_script(self) -> str:
        """Generate a SLURM array job script for this sweep."""
        return generate_sweep_script(self)

    def submit(self, dry_run: bool = False) -> str:
        """Generate and submit the sweep.

        Args:
            dry_run: If True, return the script without submitting.

        Returns:
            SLURM array job ID, or script content if dry_run.
        """
        return submit_sweep(self, dry_run=dry_run)


def generate_sweep_script(sweep: SweepConfig) -> str:
    """Generate a SLURM array job script for a parameter sweep.

    Follows the pattern from OpenEEG-Bench's job_sweep.slurm:
    defines an EXPERIMENTS array, indexes by SLURM_ARRAY_TASK_ID,
    and runs the parameterized command.
    """
    base = sweep.base
    cfg = get_cluster(base.cluster)
    if cfg.slurm is None:
        raise ValueError(f"Cluster '{cfg.name}' has no SLURM configuration")

    slurm = cfg.slurm
    time_limit = base.time_limit or slurm.time_limit
    partition = base.partition or slurm.partition
    cpus = base.cpus_per_task or slurm.cpus_per_task
    mem = base.mem_gb or slurm.mem_gb

    combos = sweep.combinations
    n_jobs = len(combos)
    param_keys = list(sweep.parameters.keys())

    # Build the EXPERIMENTS array entries
    experiment_entries: list[str] = []
    for combo in combos:
        entry = ",".join(combo[k] for k in param_keys)
        experiment_entries.append(f'    "{entry}"')
    experiments_block = "\n".join(experiment_entries)

    # Build IFS read statement
    read_vars = " ".join(k.upper() for k in param_keys)
    ifs_line = f'IFS=\',\' read -r {read_vars} <<< "${{EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}}"'

    # Build command with uppercase variable substitution
    command = sweep.command_template
    for key in param_keys:
        command = command.replace(f"{{{key}}}", f"${key.upper()}")

    preamble = _build_preamble(base, cfg)

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={base.name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks-per-node={slurm.tasks_per_node}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --gpus={base.gpus}",
        f"#SBATCH --mem={mem}G",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --array=0-{n_jobs - 1}%{sweep.max_concurrent}",
        f"#SBATCH --output={base.log_dir}/{base.name}_%A_%a.out",
        f"#SBATCH --error={base.log_dir}/{base.name}_%A_%a.err",
    ]

    if slurm.qos:
        lines.append(f"#SBATCH --qos={slurm.qos}")

    lines.append("")
    lines.append(preamble)
    lines.append("")
    lines.append(f"mkdir -p {base.log_dir}")
    lines.append("")
    lines.append(f"# ── Parameter grid ({n_jobs} combinations) ──")
    lines.append("EXPERIMENTS=(")
    lines.append(experiments_block)
    lines.append(")")
    lines.append("")
    lines.append("# Parse parameters for this array task")
    lines.append(ifs_line)
    lines.append("")
    lines.append(f'echo "=== Sweep task $SLURM_ARRAY_TASK_ID / {n_jobs - 1} ==="')
    echo_params = ", ".join(f"{k}=${k.upper()}" for k in param_keys)
    lines.append(f'echo "Parameters: {echo_params}"')
    lines.append("")
    lines.append(command)

    return "\n".join(lines) + "\n"


def submit_sweep(sweep: SweepConfig, dry_run: bool = False) -> str:
    """Generate and submit a sweep as a SLURM array job via SSH.

    Args:
        sweep: Sweep configuration.
        dry_run: If True, return script without submitting.

    Returns:
        SLURM array job ID, or script content if dry_run.
    """
    script = generate_sweep_script(sweep)

    if dry_run:
        return script

    base = sweep.base
    cfg = get_cluster(base.cluster)

    # Ensure log dir exists
    ssh_run(cfg, f"mkdir -p {base.repo_path}/{base.log_dir}", check=False)

    # Pipe to sbatch via stdin
    result = ssh_run(cfg, "sbatch", stdin=script)
    output = result.stdout.strip()
    if "Submitted" in output:
        return output.split()[-1]
    return output

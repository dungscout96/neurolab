"""Parameter sweep generation and submission.

Creates SLURM array jobs from combinatorial parameter grids,
following the pattern used in OpenEEG-Bench's sweep scripts.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

from neurolab.config import get_cluster
from neurolab.jobs.submit import SlurmJob, submit_job


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep.

    Generates all combinations from the parameter grid and creates
    either a SLURM array job or individual jobs.

    Attributes:
        name: Sweep name (used for job naming and log directories).
        command_template: Command with {param_name} placeholders.
        parameters: Dict of parameter names to lists of values.
        cluster: Target cluster name.
        max_concurrent: Maximum simultaneous array tasks (%N in --array).
        time_limit: Per-job time limit override.
        gpus: GPUs per job.
        extra_setup: Additional shell commands before the main command.
        env_overrides: Env vars to override cluster defaults.
    """

    name: str
    command_template: str
    parameters: dict[str, list[str]]
    cluster: str = "expanse"
    max_concurrent: int = 4
    time_limit: str | None = None
    gpus: int = 1
    extra_setup: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def combinations(self) -> list[dict[str, str]]:
        """Generate all parameter combinations."""
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @property
    def n_jobs(self) -> int:
        return len(self.combinations)


def generate_sweep_script(sweep: SweepConfig) -> str:
    """Generate a SLURM array job script for a parameter sweep.

    Follows the pattern from OpenEEG-Bench's job_sweep.slurm:
    defines an EXPERIMENTS array, indexes by SLURM_ARRAY_TASK_ID,
    and runs the parameterized command.

    Args:
        sweep: Sweep configuration.

    Returns:
        Complete SLURM script as a string.
    """
    cfg = get_cluster(sweep.cluster)
    if cfg.slurm is None:
        raise ValueError(f"Cluster '{cfg.name}' has no SLURM configuration")

    slurm = cfg.slurm
    time_limit = sweep.time_limit or slurm.time_limit
    combos = sweep.combinations
    n_jobs = len(combos)
    param_keys = list(sweep.parameters.keys())

    # Build the EXPERIMENTS array
    # Each entry is "val1,val2,val3" for easy IFS splitting
    experiment_entries: list[str] = []
    for combo in combos:
        entry = ",".join(combo[k] for k in param_keys)
        experiment_entries.append(f'    "{entry}"')

    experiments_block = "\n".join(experiment_entries)

    # Build IFS read statement
    read_vars = " ".join(k.upper() for k in param_keys)
    ifs_line = f"IFS=',' read -r {read_vars} <<< \"${{EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}}\""

    # Build command with uppercase variable substitution
    command = sweep.command_template
    for key in param_keys:
        command = command.replace(f"{{{key}}}", f"${key.upper()}")

    # Extra setup commands
    extra_setup = "\n".join(sweep.extra_setup) if sweep.extra_setup else ""

    # Env var exports for the script body (in addition to SBATCH --export)
    env_lines = ""
    if sweep.env_overrides:
        env_lines = "\n".join(
            f'export {k}="{v}"' for k, v in sorted(sweep.env_overrides.items())
        )

    script = dedent(f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name={sweep.name}
        #SBATCH --partition={slurm.partition}
        #SBATCH --account={slurm.account}
        #SBATCH --nodes={slurm.nodes}
        #SBATCH --ntasks-per-node={slurm.tasks_per_node}
        #SBATCH --cpus-per-task={slurm.cpus_per_task}
        #SBATCH --gpus={sweep.gpus}
        #SBATCH --mem={slurm.mem_gb}G
        #SBATCH --time={time_limit}
        #SBATCH --array=0-{n_jobs - 1}%{sweep.max_concurrent}
        #SBATCH --output=logs/{sweep.name}_%A_%a.out
        #SBATCH --error=logs/{sweep.name}_%A_%a.err

        # ── Environment setup ──
        {cfg.render_env_setup()}

        {extra_setup}
        {env_lines}

        # ── Parameter grid ({n_jobs} combinations) ──
        EXPERIMENTS=(
        {experiments_block}
        )

        # Parse parameters for this array task
        {ifs_line}

        echo "=== Sweep task $SLURM_ARRAY_TASK_ID / {n_jobs - 1} ==="
        echo "Parameters: {', '.join(f'{k}=${k.upper()}' for k in param_keys)}"
        echo ""

        # ── Run ──
        {command}
    """)

    return script


def submit_sweep(
    sweep: SweepConfig,
    dry_run: bool = False,
    script_dir: str | Path = "slurm/generated",
    ssh: bool = False,
) -> str:
    """Generate and submit a sweep as a SLURM array job.

    Args:
        sweep: Sweep configuration.
        dry_run: If True, write script but don't submit.
        script_dir: Directory to write the generated script.
        ssh: If True, submit via SSH.

    Returns:
        The SLURM job array ID, or script path if dry_run.
    """
    script_content = generate_sweep_script(sweep)

    out_dir = Path(script_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / f"{sweep.name}.slurm"
    script_path.write_text(script_content)

    if dry_run:
        return str(script_path)

    # Submit the array job
    from neurolab.jobs.submit import submit_script

    return submit_script(script_path, cluster=sweep.cluster, ssh=ssh)

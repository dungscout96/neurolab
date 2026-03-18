"""Verify environment switching between jamming, expanse, and local.

Demonstrates that the same logical path resolves differently on each
cluster, and that job scripts are generated correctly per environment.
"""

from neurolab.config import get_cluster, list_clusters
from neurolab.data import resolve_data_path, resolve_results_path
from neurolab.jobs import SlurmJob
from neurolab.jobs.submit import _render_job_script


def show_cluster(name: str) -> None:
    """Print cluster config summary."""
    c = get_cluster(name)
    print(f"\n{'=' * 60}")
    print(f"Cluster: {c.name}")
    print(f"{'=' * 60}")
    print(f"  Scheduler:    {c.scheduler}")
    print(f"  Login host:   {c.login_host or '(local)'}")
    print(f"  Is HPC:       {c.is_hpc}")
    print(f"  Data root:    {c.paths.data}")
    print(f"  Results root: {c.paths.results}")
    print(f"  Conda env:    {c.conda_env or '(active env)'}")
    print(f"  Modules:      {c.modules or '(none)'}")
    print(f"  Env vars:     {dict(c.env_vars)}")
    if c.slurm:
        print(f"  SLURM:        partition={c.slurm.partition}, account={c.slurm.account}")
    print()

    # Resolve the same logical path on each cluster
    data = resolve_data_path("hbn/preprocessed/R1/ThePresent", cluster=name)
    results = resolve_results_path("experiment_001", cluster=name)
    print(f"  Data path:    {data}")
    print(f"  Results path: {results}")


def main():
    print(f"Available clusters: {list_clusters()}")

    # Show all three environments
    for name in ["jamming", "expanse", "local"]:
        show_cluster(name)

    # Generate a training job for Expanse (SLURM)
    print(f"\n{'=' * 60}")
    print("SLURM job script for EXPANSE:")
    print(f"{'=' * 60}")
    job_expanse = SlurmJob(
        name="train_labram_lora",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        cluster="expanse",
        time_limit="04:00:00",
        gpus=1,
        working_dir="/expanse/projects/nemar/adapter_finetuning",
    )
    print(_render_job_script(job_expanse))

    # Show what jamming would look like (direct execution, no SLURM)
    print(f"\n{'=' * 60}")
    print("JAMMING direct execution command:")
    print(f"{'=' * 60}")
    jamming = get_cluster("jamming")
    cmd = "python scripts/train.py adapter=lora model=labram data=bcic2a"
    print(f"  ssh {jamming.login_host} '{cmd}'")
    print(f"  (No SLURM — direct GPU execution)")


if __name__ == "__main__":
    main()

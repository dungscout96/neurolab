"""Verify environment switching between jamming, expanse, and local.

Demonstrates that job scripts are generated correctly per environment,
with different SLURM settings, modules, and env vars.
"""

from neurolab.jobs import get_cluster, list_clusters, Job
from neurolab.jobs.submit import _render_slurm_script, _render_direct_script


def show_cluster(name: str) -> None:
    """Print cluster config summary."""
    c = get_cluster(name)
    print(f"\n{'=' * 60}")
    print(f"Cluster: {c.name}")
    print(f"{'=' * 60}")
    print(f"  Scheduler:    {c.scheduler}")
    print(f"  Login host:   {c.login_host or '(local)'}")
    print(f"  SSH target:   {c.ssh_target or '(none)'}")
    print(f"  Is HPC:       {c.is_hpc}")
    print(f"  Conda env:    {c.conda_env or '(active env)'}")
    print(f"  Modules:      {c.modules or '(none)'}")
    print(f"  Env vars:     {dict(c.env_vars)}")
    if c.slurm:
        print(f"  SLURM:        partition={c.slurm.partition}, account={c.slurm.account}")
    print()


def main():
    print(f"Available clusters: {list_clusters()}")

    # Show all three environments
    for name in ["jamming", "expanse", "local"]:
        show_cluster(name)

    # Generate a training job for Expanse (SLURM)
    print(f"\n{'=' * 60}")
    print("SLURM job script for EXPANSE:")
    print(f"{'=' * 60}")
    job_expanse = Job(
        name="train_labram_lora",
        cluster="expanse",
        repo_path="/expanse/projects/nemar/adapter_finetuning",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
        time_limit="04:00:00",
        gpus=1,
    )
    cfg_expanse = get_cluster("expanse")
    print(_render_slurm_script(job_expanse, cfg_expanse))

    # Show what jamming would look like (direct execution, no SLURM)
    print(f"\n{'=' * 60}")
    print("JAMMING direct execution script:")
    print(f"{'=' * 60}")
    job_jamming = Job(
        name="train_labram_lora",
        cluster="jamming",
        repo_path="~/projects/OpenEEG-Bench",
        command="python scripts/train.py adapter=lora model=labram data=bcic2a",
    )
    cfg_jamming = get_cluster("jamming")
    print(_render_direct_script(job_jamming, cfg_jamming))
    print(f"  (No SLURM — direct GPU execution via ssh {cfg_jamming.ssh_target})")


if __name__ == "__main__":
    main()

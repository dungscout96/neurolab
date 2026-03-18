"""Example: Using neurolab-config for cluster-aware configuration.

Shows how to auto-detect the cluster, access paths and env vars,
and generate environment setup scripts.
"""

from neurolab.config import (
    auto_detect_cluster,
    get_cluster,
    list_clusters,
    EnvironmentManager,
)


def main():
    # List available cluster profiles
    print("Available clusters:", list_clusters())

    # Auto-detect current cluster (uses hostname + NEUROLAB_CLUSTER env var)
    cluster = auto_detect_cluster()
    print(f"\nDetected cluster: {cluster.name}")
    print(f"  Data root:  {cluster.paths.data}")
    print(f"  Results:    {cluster.paths.results}")
    print(f"  Is HPC:     {cluster.is_hpc}")

    # Explicit cluster selection
    expanse = get_cluster("expanse")
    print(f"\nExpanse config:")
    print(f"  Login host: {expanse.login_host}")
    print(f"  Partition:  {expanse.slurm.partition}")
    print(f"  Account:    {expanse.slurm.account}")
    print(f"  Modules:    {expanse.modules}")
    print(f"  HF_HOME:    {expanse.env_vars.get('HF_HOME')}")

    # Generate shell setup script
    print("\n--- Expanse environment setup script ---")
    print(expanse.render_env_setup())

    # Environment manager: apply env vars to current process
    env = EnvironmentManager(cluster)
    warnings = env.validate()
    if warnings:
        print("\nValidation warnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()

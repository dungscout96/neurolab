"""neurolab-config: Cluster environment configuration for EEG research infrastructure.

Provides unified configuration across local development and HPC clusters
(Expanse, Delta) including environment modules, conda environments, paths,
and authentication credentials.

Usage:
    from neurolab.config import get_cluster, auto_detect_cluster

    cluster = auto_detect_cluster()          # Detects from hostname/env
    cluster = get_cluster("expanse")         # Explicit selection

    print(cluster.name)                      # "expanse"
    print(cluster.paths.data)                # "/expanse/projects/nemar/dtyoung/data"
    print(cluster.conda_env)                 # "/expanse/projects/nemar/dtyoung/conda_envs/..."
    print(cluster.env_vars["HF_HOME"])       # "/expanse/projects/nemar/dtyoung/huggingface_cache"

    env_script = cluster.render_env_setup()  # Shell script for module loads + exports
"""

from neurolab.config.cluster import (
    ClusterConfig,
    ClusterPaths,
    SlurmDefaults,
    auto_detect_cluster,
    get_cluster,
    list_clusters,
    register_cluster_profile,
)
from neurolab.config.environment import EnvironmentManager

__all__ = [
    "ClusterConfig",
    "ClusterPaths",
    "SlurmDefaults",
    "auto_detect_cluster",
    "get_cluster",
    "list_clusters",
    "register_cluster_profile",
    "EnvironmentManager",
]

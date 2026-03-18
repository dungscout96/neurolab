"""Cluster-aware path resolution.

Translates logical dataset paths to absolute filesystem paths based on
the current (or specified) cluster. This is the key abstraction that
lets the same code run on local, Expanse, and Delta without changes.

The path resolution follows a simple rule:
    absolute_path = cluster.paths.<root> / relative_path

where <root> is one of: data, scratch, results, cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from neurolab.config import ClusterConfig, auto_detect_cluster, get_cluster


def _get_cluster(cluster: str | ClusterConfig | None) -> ClusterConfig:
    """Resolve a cluster argument to a ClusterConfig."""
    if cluster is None:
        return auto_detect_cluster()
    if isinstance(cluster, str):
        return get_cluster(cluster)
    return cluster


def resolve_data_path(
    relative: str,
    cluster: str | ClusterConfig | None = None,
    mkdir: bool = False,
) -> Path:
    """Resolve a relative path against the cluster's data root.

    Args:
        relative: Relative path within the data directory
            (e.g. "hbn/preprocessed/R1/ThePresent").
        cluster: Cluster name, config, or None for auto-detect.
        mkdir: If True, create the directory if it doesn't exist.

    Returns:
        Absolute Path on the current filesystem.

    Examples:
        # On Expanse:
        resolve_data_path("hbn/preprocessed/R1")
        → Path("/expanse/projects/nemar/dtyoung/data/hbn/preprocessed/R1")

        # Local:
        resolve_data_path("hbn/preprocessed/R1")
        → Path("/home/user/data/eeg/hbn/preprocessed/R1")
    """
    cfg = _get_cluster(cluster)
    base = Path(os.path.expanduser(cfg.paths.data))
    result = base / relative

    if mkdir:
        result.mkdir(parents=True, exist_ok=True)

    return result


def resolve_scratch_path(
    relative: str,
    cluster: str | ClusterConfig | None = None,
    mkdir: bool = False,
) -> Path:
    """Resolve a relative path against the cluster's scratch space.

    Scratch is for temporary/intermediate files that don't need
    long-term persistence.
    """
    cfg = _get_cluster(cluster)
    base = Path(os.path.expanduser(cfg.paths.scratch))
    result = base / relative

    if mkdir:
        result.mkdir(parents=True, exist_ok=True)

    return result


def resolve_results_path(
    relative: str,
    cluster: str | ClusterConfig | None = None,
    mkdir: bool = False,
) -> Path:
    """Resolve a relative path against the cluster's results directory.

    Results contain experiment outputs, checkpoints, and logs.
    """
    cfg = _get_cluster(cluster)
    base = Path(os.path.expanduser(cfg.paths.results))
    result = base / relative

    if mkdir:
        result.mkdir(parents=True, exist_ok=True)

    return result


def resolve_cache_path(
    relative: str,
    cluster: str | ClusterConfig | None = None,
    mkdir: bool = False,
) -> Path:
    """Resolve a relative path against the cluster's cache directory.

    Cache is for HuggingFace models, MNE data, and other downloaded resources.
    """
    cfg = _get_cluster(cluster)
    base = Path(os.path.expanduser(cfg.paths.cache))
    result = base / relative

    if mkdir:
        result.mkdir(parents=True, exist_ok=True)

    return result

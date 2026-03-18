"""Tests for cluster configuration (merged into neurolab-jobs)."""

import os
import tempfile
from pathlib import Path

import pytest

from neurolab.jobs import (
    ClusterConfig,
    ClusterPaths,
    SlurmDefaults,
    get_cluster,
    list_clusters,
    auto_detect_cluster,
    register_cluster_profile,
    EnvironmentManager,
)


def test_list_clusters():
    """Built-in profiles should include local, expanse, delta."""
    clusters = list_clusters()
    assert "local" in clusters
    assert "expanse" in clusters
    assert "delta" in clusters


def test_get_cluster_local():
    """Local cluster should not have SLURM."""
    local = get_cluster("local")
    assert local.name == "local"
    assert local.slurm is None
    assert local.is_hpc is False
    assert local.scheduler == "local"


def test_get_cluster_expanse():
    """Expanse should have SLURM with gpu-shared partition."""
    expanse = get_cluster("expanse")
    assert expanse.name == "expanse"
    assert expanse.is_hpc is True
    assert expanse.slurm is not None
    assert expanse.slurm.partition == "gpu-shared"
    assert expanse.slurm.account == "csd403"
    assert "gpu" in expanse.modules
    assert "HF_HOME" in expanse.env_vars


def test_get_cluster_delta():
    """Delta should have SLURM with gpuA100x4 partition."""
    delta = get_cluster("delta")
    assert delta.name == "delta"
    assert delta.is_hpc is True
    assert delta.slurm is not None
    assert "A100" in delta.slurm.partition


def test_get_cluster_unknown():
    """Unknown cluster name should raise KeyError."""
    with pytest.raises(KeyError, match="Unknown cluster"):
        get_cluster("nonexistent_cluster")


def test_auto_detect_fallback():
    """Auto-detect should fall back to local on unknown hosts."""
    # Remove explicit override if set
    old = os.environ.pop("NEUROLAB_CLUSTER", None)
    try:
        cluster = auto_detect_cluster()
        assert cluster.name == "local"
    finally:
        if old:
            os.environ["NEUROLAB_CLUSTER"] = old


def test_auto_detect_explicit():
    """NEUROLAB_CLUSTER env var should override auto-detection."""
    os.environ["NEUROLAB_CLUSTER"] = "expanse"
    try:
        cluster = auto_detect_cluster()
        assert cluster.name == "expanse"
    finally:
        del os.environ["NEUROLAB_CLUSTER"]


def test_render_env_setup():
    """render_env_setup should produce valid shell commands."""
    expanse = get_cluster("expanse")
    script = expanse.render_env_setup()
    assert "module purge" in script
    assert "module load gpu" in script
    assert "source activate" in script
    assert "export HF_HOME=" in script


def test_register_custom_profile():
    """Should be able to register a custom cluster profile."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
name: test_cluster
hostname_patterns: [testhost]
scheduler: slurm
paths:
  data: /test/data
  scratch: /test/scratch
  results: /test/results
  conda_envs: /test/conda
  cache: /test/cache
slurm:
  partition: test-gpu
  account: test-account
modules: [cuda/11.0]
conda_env: /test/conda/myenv
env_vars:
  TEST_VAR: hello
""")
        f.flush()
        cfg = register_cluster_profile(f.name)

    assert cfg.name == "test_cluster"
    assert cfg.slurm.partition == "test-gpu"
    assert cfg.env_vars["TEST_VAR"] == "hello"
    os.unlink(f.name)


def test_cluster_paths_resolve():
    """ClusterPaths.resolve should join subpath to data root."""
    paths = ClusterPaths(
        data="/data/root",
        scratch="/scratch",
        results="/results",
        conda_envs="/conda",
        cache="/cache",
    )
    assert paths.resolve("hbn/R1") == "/data/root/hbn/R1"


def test_environment_manager_apply():
    """EnvironmentManager.apply should set env vars."""
    local = get_cluster("local")
    env = EnvironmentManager(local)

    # Remove the var first
    old = os.environ.pop("WANDB_MODE", None)
    try:
        env.apply()
        assert "WANDB_MODE" in os.environ
    finally:
        if old:
            os.environ["WANDB_MODE"] = old
        else:
            os.environ.pop("WANDB_MODE", None)

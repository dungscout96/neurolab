"""Cluster configuration management.

Loads YAML profiles for each compute environment and provides a unified
interface to access paths, SLURM defaults, environment variables, and
module loading commands.
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterPaths:
    """Filesystem paths for a cluster environment."""

    data: str
    """Root directory for datasets (raw and preprocessed)."""

    scratch: str
    """Fast scratch space for intermediate/temporary files."""

    results: str
    """Directory for experiment outputs, checkpoints, and logs."""

    conda_envs: str
    """Base directory for conda environments."""

    cache: str
    """Cache directory (HuggingFace, MNE, etc.)."""

    def resolve(self, subpath: str) -> str:
        """Resolve a subpath relative to the data root."""
        return str(Path(self.data) / subpath)


@dataclass(frozen=True)
class SlurmDefaults:
    """Default SLURM resource requests for a cluster."""

    partition: str
    account: str
    gpus_per_node: int = 1
    cpus_per_task: int = 8
    mem_gb: int = 64
    time_limit: str = "02:00:00"
    nodes: int = 1
    tasks_per_node: int = 1
    qos: str | None = None


@dataclass
class ClusterConfig:
    """Complete configuration for a compute environment.

    Encapsulates everything needed to run experiments on a given cluster:
    filesystem paths, SLURM resource defaults, module loading commands,
    conda environment activation, and environment variable exports.
    """

    name: str
    """Identifier for this cluster (e.g. 'expanse', 'delta', 'local')."""

    hostname_patterns: list[str]
    """Hostname substrings used for auto-detection (e.g. ['expanse', 'tscc'])."""

    paths: ClusterPaths
    """Filesystem paths for this environment."""

    slurm: SlurmDefaults | None = None
    """Default SLURM parameters. None for local-only environments."""

    modules: list[str] = field(default_factory=list)
    """Module load commands (e.g. ['gpu', 'cuda12.2/toolkit/12.2.2'])."""

    conda_env: str = ""
    """Full path to the conda environment to activate."""

    env_vars: dict[str, str] = field(default_factory=dict)
    """Environment variables to export (HF_HOME, WANDB_MODE, etc.)."""

    login_host: str = ""
    """SSH login hostname (e.g. 'login.expanse.sdsc.edu')."""

    scheduler: str = "slurm"
    """Job scheduler type. 'slurm' for HPC, 'local' for direct execution."""

    @property
    def is_hpc(self) -> bool:
        """Whether this is an HPC cluster (has SLURM)."""
        return self.slurm is not None

    def get_env_var(self, key: str, default: str = "") -> str:
        """Get an environment variable, checking config then actual env."""
        return self.env_vars.get(key, os.environ.get(key, default))

    def render_env_setup(self) -> str:
        """Render a shell script fragment for environment setup.

        This produces the preamble used in SLURM job scripts or interactive
        sessions: module loads, conda activation, and env var exports.
        """
        lines: list[str] = []

        # Module loads
        if self.modules:
            lines.append("module purge")
            for mod in self.modules:
                lines.append(f"module load {mod}")
            lines.append("")

        # Conda activation
        if self.conda_env:
            lines.append(f"source activate {self.conda_env}")
            lines.append("")

        # Environment variables
        for key, value in sorted(self.env_vars.items()):
            lines.append(f'export {key}="{value}"')

        return "\n".join(lines)

    def render_slurm_exports(self) -> str:
        """Render env vars as a comma-separated SBATCH --export string."""
        if not self.env_vars:
            return ""
        pairs = [f"{k}={v}" for k, v in sorted(self.env_vars.items())]
        return ",".join(pairs)


# ── Profile registry ────────────────────────────────────────────────

_PROFILES_DIR = Path(__file__).parent / "profiles"
_registry: dict[str, ClusterConfig] = {}


def _load_yaml_profile(path: Path) -> ClusterConfig:
    """Parse a YAML profile file into a ClusterConfig."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Resolve .env file for secrets if specified
    env_file = raw.get("env_file")
    extra_env: dict[str, str] = {}
    if env_file:
        env_path = Path(env_file).expanduser()
        if env_path.exists():
            extra_env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}

    paths_raw = raw.get("paths", {})
    paths = ClusterPaths(
        data=paths_raw.get("data", ""),
        scratch=paths_raw.get("scratch", ""),
        results=paths_raw.get("results", ""),
        conda_envs=paths_raw.get("conda_envs", ""),
        cache=paths_raw.get("cache", ""),
    )

    slurm_raw = raw.get("slurm")
    slurm = None
    if slurm_raw:
        slurm = SlurmDefaults(
            partition=slurm_raw.get("partition", ""),
            account=slurm_raw.get("account", ""),
            gpus_per_node=slurm_raw.get("gpus_per_node", 1),
            cpus_per_task=slurm_raw.get("cpus_per_task", 8),
            mem_gb=slurm_raw.get("mem_gb", 64),
            time_limit=slurm_raw.get("time_limit", "02:00:00"),
            nodes=slurm_raw.get("nodes", 1),
            tasks_per_node=slurm_raw.get("tasks_per_node", 1),
            qos=slurm_raw.get("qos"),
        )

    # Merge env vars from profile and .env file (profile takes precedence)
    env_vars = {**extra_env, **raw.get("env_vars", {})}

    return ClusterConfig(
        name=raw.get("name", path.stem),
        hostname_patterns=raw.get("hostname_patterns", []),
        paths=paths,
        slurm=slurm,
        modules=raw.get("modules", []),
        conda_env=raw.get("conda_env", ""),
        env_vars=env_vars,
        login_host=raw.get("login_host", ""),
        scheduler=raw.get("scheduler", "slurm" if slurm_raw else "local"),
    )


def _ensure_loaded() -> None:
    """Lazily load all built-in profiles on first access."""
    if _registry:
        return
    if _PROFILES_DIR.is_dir():
        for yaml_file in sorted(_PROFILES_DIR.glob("*.yaml")):
            cfg = _load_yaml_profile(yaml_file)
            _registry[cfg.name] = cfg


def register_cluster_profile(path: str | Path) -> ClusterConfig:
    """Register a custom cluster profile from a YAML file.

    Use this to add project-specific or user-specific cluster configs
    beyond the built-in local/expanse/delta profiles.

    Args:
        path: Path to a YAML profile file.

    Returns:
        The loaded ClusterConfig.
    """
    cfg = _load_yaml_profile(Path(path))
    _registry[cfg.name] = cfg
    return cfg


def get_cluster(name: str) -> ClusterConfig:
    """Get a cluster configuration by name.

    Args:
        name: Cluster identifier (e.g. 'local', 'expanse', 'delta').

    Raises:
        KeyError: If no profile is registered with that name.
    """
    _ensure_loaded()
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise KeyError(f"Unknown cluster '{name}'. Available: {available}")
    return _registry[name]


def list_clusters() -> list[str]:
    """List all registered cluster names."""
    _ensure_loaded()
    return sorted(_registry.keys())


def auto_detect_cluster() -> ClusterConfig:
    """Detect the current cluster from hostname and environment.

    Detection order:
    1. NEUROLAB_CLUSTER environment variable (explicit override)
    2. Hostname substring matching against registered profiles
    3. Falls back to 'local'

    Returns:
        The detected ClusterConfig.
    """
    _ensure_loaded()

    # 1. Explicit override
    explicit = os.environ.get("NEUROLAB_CLUSTER")
    if explicit:
        return get_cluster(explicit)

    # 2. Hostname matching
    hostname = socket.gethostname().lower()
    for cfg in _registry.values():
        for pattern in cfg.hostname_patterns:
            if pattern.lower() in hostname:
                return cfg

    # 3. Fallback to local
    if "local" in _registry:
        return _registry["local"]

    raise RuntimeError(
        "Could not auto-detect cluster and no 'local' profile found. "
        "Set NEUROLAB_CLUSTER env var or register a profile."
    )

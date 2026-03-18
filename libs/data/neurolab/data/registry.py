"""Dataset registry.

A central registry that maps logical dataset names to their relative
paths and metadata. This decouples experiment scripts from filesystem
details — you reference "hbn_R1_ThePresent" and the registry + path
resolver handles the rest.

Registries can be built programmatically or loaded from YAML files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neurolab.config import ClusterConfig
from neurolab.data.paths import resolve_data_path


@dataclass
class DatasetEntry:
    """Metadata for a registered dataset.

    Attributes:
        name: Logical name (e.g. "hbn_R1_ThePresent").
        relative_path: Path relative to the cluster's data root.
        description: Human-readable description.
        format: Data format (e.g. "fif", "bids", "eeglab", "npy").
        splits: Named splits with their own relative sub-paths.
        metadata: Arbitrary extra metadata.
    """

    name: str
    relative_path: str
    description: str = ""
    format: str = ""
    splits: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self, cluster: str | ClusterConfig | None = None) -> Path:
        """Resolve this dataset's absolute path on the given cluster."""
        return resolve_data_path(self.relative_path, cluster=cluster)

    def resolve_split(
        self, split: str, cluster: str | ClusterConfig | None = None
    ) -> Path:
        """Resolve the absolute path for a named split.

        Args:
            split: Split name (e.g. "train", "val", "test").
            cluster: Cluster to resolve paths for.

        Raises:
            KeyError: If the split is not defined.
        """
        if split not in self.splits:
            available = ", ".join(sorted(self.splits.keys()))
            raise KeyError(
                f"Split '{split}' not found for dataset '{self.name}'. "
                f"Available: {available}"
            )
        return resolve_data_path(self.splits[split], cluster=cluster)


class DataRegistry:
    """Central registry mapping dataset names to paths and metadata.

    Usage:
        registry = DataRegistry()

        # Register datasets programmatically
        registry.register(
            name="hbn_R1_ThePresent",
            relative_path="hbn/preprocessed/R1/ThePresent",
            format="fif",
            splits={
                "train": "hbn/preprocessed/R1/ThePresent",
                "val": "hbn/preprocessed/R5/ThePresent",
                "test": "hbn/preprocessed/R6/ThePresent",
            },
        )

        # Or load from YAML
        registry = DataRegistry.from_yaml("datasets.yaml")

        # Use
        path = registry.resolve("hbn_R1_ThePresent")
        train_path = registry.resolve_split("hbn_R1_ThePresent", "train")
    """

    def __init__(self) -> None:
        self._entries: dict[str, DatasetEntry] = {}

    def register(
        self,
        name: str,
        relative_path: str,
        description: str = "",
        format: str = "",
        splits: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetEntry:
        """Register a dataset.

        Args:
            name: Logical name for the dataset.
            relative_path: Path relative to cluster data root.
            description: Human-readable description.
            format: Data format identifier.
            splits: Optional dict of split_name → relative_path.
            metadata: Arbitrary extra metadata.

        Returns:
            The created DatasetEntry.
        """
        entry = DatasetEntry(
            name=name,
            relative_path=relative_path,
            description=description,
            format=format,
            splits=splits or {},
            metadata=metadata or {},
        )
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> DatasetEntry:
        """Get a dataset entry by name.

        Raises:
            KeyError: If the dataset is not registered.
        """
        if name not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return self._entries[name]

    def resolve(
        self, name: str, cluster: str | ClusterConfig | None = None
    ) -> Path:
        """Resolve a dataset's absolute path."""
        return self.get(name).resolve(cluster=cluster)

    def resolve_split(
        self,
        name: str,
        split: str,
        cluster: str | ClusterConfig | None = None,
    ) -> Path:
        """Resolve a dataset split's absolute path."""
        return self.get(name).resolve_split(split, cluster=cluster)

    def list_datasets(self) -> list[str]:
        """List all registered dataset names."""
        return sorted(self._entries.keys())

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataRegistry:
        """Load a registry from a YAML file.

        Expected format:
            datasets:
              hbn_R1_ThePresent:
                relative_path: hbn/preprocessed/R1/ThePresent
                format: fif
                description: HBN Release 1 - ThePresent task
                splits:
                  train: hbn/preprocessed/R1/ThePresent
                  val: hbn/preprocessed/R5/ThePresent
                  test: hbn/preprocessed/R6/ThePresent
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        registry = cls()
        for name, entry_data in raw.get("datasets", {}).items():
            registry.register(
                name=name,
                relative_path=entry_data["relative_path"],
                description=entry_data.get("description", ""),
                format=entry_data.get("format", ""),
                splits=entry_data.get("splits", {}),
                metadata=entry_data.get("metadata", {}),
            )
        return registry

    def to_yaml(self, path: str | Path) -> None:
        """Save the registry to a YAML file."""
        data = {"datasets": {}}
        for name, entry in sorted(self._entries.items()):
            d: dict[str, Any] = {"relative_path": entry.relative_path}
            if entry.description:
                d["description"] = entry.description
            if entry.format:
                d["format"] = entry.format
            if entry.splits:
                d["splits"] = entry.splits
            if entry.metadata:
                d["metadata"] = entry.metadata
            data["datasets"][name] = d

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

"""Tests for neurolab-data."""

import os
import tempfile
from pathlib import Path

import pytest

from neurolab.data import resolve_data_path, resolve_results_path, DataRegistry


def test_resolve_data_path_local():
    """Data path resolution should use cluster's data root."""
    old = os.environ.pop("NEUROLAB_CLUSTER", None)
    try:
        path = resolve_data_path("hbn/preprocessed/R1", cluster="local")
        assert str(path).endswith("hbn/preprocessed/R1")
    finally:
        if old:
            os.environ["NEUROLAB_CLUSTER"] = old


def test_resolve_data_path_expanse():
    """Expanse data paths should use /expanse/projects/nemar."""
    path = resolve_data_path("hbn/preprocessed/R1", cluster="expanse")
    assert "/expanse/projects/nemar" in str(path)
    assert str(path).endswith("hbn/preprocessed/R1")


def test_resolve_data_path_delta():
    """Delta data paths should use /projects/bcfj."""
    path = resolve_data_path("hbn/preprocessed/R1", cluster="delta")
    assert "/projects/bcfj" in str(path)
    assert str(path).endswith("hbn/preprocessed/R1")


def test_resolve_results_path():
    """Results path should resolve correctly."""
    path = resolve_results_path("experiment_001", cluster="expanse")
    assert "/expanse/projects/nemar" in str(path)
    assert str(path).endswith("experiment_001")


def test_registry_programmatic():
    """DataRegistry should support programmatic registration."""
    registry = DataRegistry()
    registry.register(
        name="test_dataset",
        relative_path="test/data",
        format="fif",
        splits={"train": "test/train", "val": "test/val"},
    )

    assert "test_dataset" in registry.list_datasets()
    entry = registry.get("test_dataset")
    assert entry.format == "fif"
    assert "train" in entry.splits


def test_registry_resolve():
    """Registry resolve should return a Path."""
    registry = DataRegistry()
    registry.register(name="test", relative_path="my/dataset")

    path = registry.resolve("test", cluster="local")
    assert isinstance(path, Path)
    assert str(path).endswith("my/dataset")


def test_registry_resolve_split():
    """Registry should resolve split paths."""
    registry = DataRegistry()
    registry.register(
        name="test",
        relative_path="my/dataset",
        splits={"train": "my/dataset/train", "val": "my/dataset/val"},
    )

    train_path = registry.resolve_split("test", "train", cluster="local")
    assert str(train_path).endswith("my/dataset/train")


def test_registry_unknown_dataset():
    """Unknown dataset should raise KeyError."""
    registry = DataRegistry()
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent")


def test_registry_unknown_split():
    """Unknown split should raise KeyError."""
    registry = DataRegistry()
    registry.register(name="test", relative_path="test", splits={"train": "test/train"})

    with pytest.raises(KeyError, match="Split 'val' not found"):
        registry.resolve_split("test", "val")


def test_registry_yaml_roundtrip():
    """Registry should survive YAML save/load."""
    registry = DataRegistry()
    registry.register(
        name="hbn_test",
        relative_path="hbn/preprocessed",
        description="Test dataset",
        format="fif",
        splits={"train": "hbn/train", "val": "hbn/val"},
        metadata={"sfreq": 200, "n_channels": 129},
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        registry.to_yaml(f.name)
        loaded = DataRegistry.from_yaml(f.name)

    assert "hbn_test" in loaded.list_datasets()
    entry = loaded.get("hbn_test")
    assert entry.format == "fif"
    assert entry.metadata["sfreq"] == 200
    assert entry.splits["train"] == "hbn/train"

    os.unlink(f.name)

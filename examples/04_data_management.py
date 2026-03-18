"""Example: Using neurolab-data for cluster-aware data management.

Shows path resolution, dataset registry usage, and how the same
code works across local dev and HPC clusters.
"""

from pathlib import Path

from neurolab.data import (
    resolve_data_path,
    resolve_results_path,
    resolve_scratch_path,
    DataRegistry,
)


def main():
    # ── Cluster-aware path resolution ───────────────────────────────
    # Same call resolves differently on each cluster:
    data_path = resolve_data_path("hbn/preprocessed/R1/ThePresent")
    print(f"Data path: {data_path}")
    # Expanse → /expanse/projects/nemar/dtyoung/data/hbn/preprocessed/R1/ThePresent
    # Delta   → /projects/bcfj/dtyoung/data/hbn/preprocessed/R1/ThePresent
    # Local   → ~/data/eeg/hbn/preprocessed/R1/ThePresent

    results_path = resolve_results_path("experiment_001/checkpoints")
    print(f"Results path: {results_path}")

    scratch_path = resolve_scratch_path("temp_preprocessing", mkdir=True)
    print(f"Scratch path: {scratch_path}")

    # ── Explicit cluster selection ──────────────────────────────────
    expanse_path = resolve_data_path("hbn/preprocessed/R1/ThePresent", cluster="expanse")
    delta_path = resolve_data_path("hbn/preprocessed/R1/ThePresent", cluster="delta")
    local_path = resolve_data_path("hbn/preprocessed/R1/ThePresent", cluster="local")
    print(f"\nExpanse: {expanse_path}")
    print(f"Delta:   {delta_path}")
    print(f"Local:   {local_path}")

    # ── Dataset registry ────────────────────────────────────────────
    # Load from the central catalog
    registry = DataRegistry.from_yaml("datasets.yaml")
    print(f"\nRegistered datasets: {registry.list_datasets()}")

    # Resolve a dataset's path
    hbn_path = registry.resolve("hbn_R1_ThePresent")
    print(f"HBN R1 ThePresent: {hbn_path}")

    # Resolve a specific split
    train_path = registry.resolve_split("hbn_R1_ThePresent", "train")
    val_path = registry.resolve_split("hbn_R1_ThePresent", "val")
    test_path = registry.resolve_split("hbn_R1_ThePresent", "test")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")

    # Access metadata
    entry = registry.get("hbn_R1_ThePresent")
    print(f"  Sfreq:    {entry.metadata.get('sfreq')} Hz")
    print(f"  Channels: {entry.metadata.get('n_channels')}")
    print(f"  Montage:  {entry.metadata.get('montage')}")


if __name__ == "__main__":
    main()

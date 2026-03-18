"""neurolab-data: Unified data management for EEG research.

Provides cluster-aware path resolution, data processing pipelines,
and dataset loading abstractions that work identically across local
development and HPC environments (Expanse, Delta).

Usage:
    from neurolab.data import DataRegistry, resolve_data_path

    # Cluster-aware path resolution
    path = resolve_data_path("hbn/preprocessed/R1/ThePresent")
    # → /expanse/projects/nemar/dtyoung/data/hbn/preprocessed/R1/ThePresent (on Expanse)
    # → ~/data/eeg/hbn/preprocessed/R1/ThePresent                           (local)

    # Dataset registry
    registry = DataRegistry()
    registry.register("hbn_R1", "hbn/preprocessed/R1/ThePresent")
    ds_path = registry.resolve("hbn_R1")

    # Processing pipelines
    from neurolab.data.processing import ProcessingPipeline, ProcessingStep
    pipeline = ProcessingPipeline(steps=[...])
    pipeline.run(input_dir=..., output_dir=...)
"""

from neurolab.data.paths import resolve_data_path, resolve_scratch_path, resolve_results_path
from neurolab.data.registry import DataRegistry, DatasetEntry
from neurolab.data.processing import ProcessingPipeline, ProcessingStep

__all__ = [
    "resolve_data_path",
    "resolve_scratch_path",
    "resolve_results_path",
    "DataRegistry",
    "DatasetEntry",
    "ProcessingPipeline",
    "ProcessingStep",
]

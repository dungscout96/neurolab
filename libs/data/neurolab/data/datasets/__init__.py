"""Built-in dataset implementations and base classes.

Provides abstract base classes for EEG datasets and concrete
implementations that can be shared across projects.
"""

from neurolab.data.datasets.base import (
    EEGDatasetConfig,
    BaseEEGDataset,
    WindowedEEGDataset,
)

__all__ = [
    "EEGDatasetConfig",
    "BaseEEGDataset",
    "WindowedEEGDataset",
]

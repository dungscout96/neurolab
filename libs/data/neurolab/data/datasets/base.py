"""Base dataset classes for EEG research.

Provides abstract base classes modeled after the patterns in:
- eb_jepa_eeg/eb_jepa/datasets/hbn.py (memory-mapped FIF loading,
  windowed access, per-channel normalization)
- OpenEEG-Bench/adapter_finetuning/ (Hydra-configurable datasets
  with braindecode integration)

These base classes handle the common concerns (path resolution,
windowing, normalization, train/val/test splitting) so that
project-specific datasets only need to implement data loading.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from neurolab.data.paths import resolve_data_path

logger = logging.getLogger(__name__)


@dataclass
class EEGDatasetConfig:
    """Configuration for an EEG dataset.

    This mirrors the pattern from OpenEEG-Bench's Hydra data configs,
    providing a structured way to specify loading, preprocessing, and
    windowing parameters.

    Attributes:
        name: Dataset identifier.
        relative_path: Path relative to cluster data root.
        sfreq: Sampling frequency in Hz.
        n_channels: Number of EEG channels.
        window_size_s: Window size in seconds.
        window_stride_s: Window stride in seconds (None = non-overlapping).
        n_classes: Number of classification targets (0 for regression/SSL).
        normalize: Whether to apply per-channel z-normalization.
        clip_std: Clip values beyond ±N standard deviations (0 = no clip).
        splits: Named data splits (e.g. {"train": [...], "val": [...], "test": [...]}).
    """

    name: str = ""
    relative_path: str = ""
    sfreq: float = 200.0
    n_channels: int = 0
    window_size_s: float = 4.0
    window_stride_s: float | None = None
    n_classes: int = 0
    normalize: bool = True
    clip_std: float = 0.0
    splits: dict[str, Any] = field(default_factory=dict)

    @property
    def window_size_samples(self) -> int:
        """Window size in samples."""
        return int(self.window_size_s * self.sfreq)

    @property
    def window_stride_samples(self) -> int:
        """Window stride in samples."""
        stride = self.window_stride_s if self.window_stride_s else self.window_size_s
        return int(stride * self.sfreq)


class BaseEEGDataset(ABC):
    """Abstract base class for EEG datasets.

    Handles path resolution, file discovery, and provides the
    interface that training loops expect. Subclass this and
    implement _load_recording() and __getitem__().

    This follows the pattern from eb_jepa_eeg's HBNDataset where
    recordings are discovered from a directory, and individual
    samples are windows into those recordings.
    """

    def __init__(
        self,
        config: EEGDatasetConfig,
        split: str = "train",
        cluster: str | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.cluster = cluster

        # Resolve the data path for the current cluster
        self.data_dir = resolve_data_path(config.relative_path, cluster=cluster)

        # Discover and load recording file list
        self._recordings: list[Path] = []
        self._setup()

    def _setup(self) -> None:
        """Discover recordings in the data directory.

        Override this if your dataset has a non-standard layout.
        Default: finds all .fif files in the data directory.
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return

        pattern = "*.fif"  # Override in subclass for other formats
        self._recordings = sorted(self.data_dir.glob(pattern))

        # Apply split filtering if configured
        split_filter = self.config.splits.get(self.split)
        if split_filter is not None:
            self._recordings = self._apply_split_filter(
                self._recordings, split_filter
            )

        logger.info(
            f"Dataset '{self.config.name}' [{self.split}]: "
            f"{len(self._recordings)} recordings from {self.data_dir}"
        )

    def _apply_split_filter(
        self,
        recordings: list[Path],
        split_filter: Any,
    ) -> list[Path]:
        """Filter recordings based on split specification.

        Override this for custom split logic. Default handles:
        - List of indices: [0, 1, 2, 3]
        - List of subject IDs: ["sub-001", "sub-002"]
        - Dict with 'subjects' key: {"subjects": ["sub-001"]}
        """
        if isinstance(split_filter, list):
            if all(isinstance(x, int) for x in split_filter):
                return [recordings[i] for i in split_filter if i < len(recordings)]
            elif all(isinstance(x, str) for x in split_filter):
                return [r for r in recordings if any(s in r.stem for s in split_filter)]

        if isinstance(split_filter, dict) and "subjects" in split_filter:
            subjects = split_filter["subjects"]
            return [r for r in recordings if any(s in r.stem for s in subjects)]

        return recordings

    @property
    def n_recordings(self) -> int:
        """Number of recordings in this split."""
        return len(self._recordings)

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples (windows) in this dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index.

        Returns:
            Dict with at minimum:
            - "eeg": numpy array of shape (n_channels, n_samples)
            - "label": target label (int for classification, float for regression)
            Optional:
            - "subject": subject identifier
            - "recording": recording identifier
            - "metadata": dict of additional info
        """
        ...


class WindowedEEGDataset(BaseEEGDataset):
    """Base class for windowed EEG datasets.

    Extends BaseEEGDataset with windowing logic: splits each recording
    into fixed-size windows with configurable stride. This follows the
    pattern from both eb_jepa_eeg (random crops) and OpenEEG-Bench
    (event-locked windows).

    Subclasses must implement:
    - _load_recording(path) → raw EEG array
    - _get_label(recording_idx, window_idx) → label
    """

    def __init__(
        self,
        config: EEGDatasetConfig,
        split: str = "train",
        cluster: str | None = None,
        preload: bool = False,
    ) -> None:
        super().__init__(config, split, cluster)
        self.preload = preload

        # Build a flat index: (recording_idx, start_sample)
        self._window_index: list[tuple[int, int]] = []
        self._recording_lengths: list[int] = []  # In samples
        self._cached_data: dict[int, np.ndarray] = {}

        self._build_window_index()

    def _build_window_index(self) -> None:
        """Build a flat index of (recording_idx, start_sample) pairs."""
        win_size = self.config.window_size_samples
        win_stride = self.config.window_stride_samples

        for rec_idx, rec_path in enumerate(self._recordings):
            n_samples = self._get_recording_length(rec_path)
            self._recording_lengths.append(n_samples)

            if n_samples < win_size:
                logger.debug(
                    f"Recording {rec_path.name} too short "
                    f"({n_samples} < {win_size} samples), skipping"
                )
                continue

            for start in range(0, n_samples - win_size + 1, win_stride):
                self._window_index.append((rec_idx, start))

            if self.preload:
                self._cached_data[rec_idx] = self._load_recording(rec_path)

        logger.info(
            f"Built window index: {len(self._window_index)} windows "
            f"from {len(self._recordings)} recordings"
        )

    def _get_recording_length(self, path: Path) -> int:
        """Get the length of a recording in samples without loading it.

        Override this for formats that support quick length queries.
        Default reads the file info via MNE.
        """
        try:
            import mne

            info = mne.io.read_info(str(path), verbose=False)
            # For raw FIF files, we can get n_times from the file
            raw = mne.io.read_raw_fif(str(path), preload=False, verbose=False)
            return raw.n_times
        except Exception:
            # Fallback: load and check
            data = self._load_recording(path)
            return data.shape[-1]

    @abstractmethod
    def _load_recording(self, path: Path) -> np.ndarray:
        """Load a recording from disk.

        Args:
            path: Path to the recording file.

        Returns:
            numpy array of shape (n_channels, n_samples).
        """
        ...

    def _get_recording_data(self, rec_idx: int) -> np.ndarray:
        """Get recording data, using cache if available."""
        if rec_idx in self._cached_data:
            return self._cached_data[rec_idx]

        data = self._load_recording(self._recordings[rec_idx])

        if self.preload:
            self._cached_data[rec_idx] = data

        return data

    def _get_label(self, recording_idx: int, window_start: int) -> Any:
        """Get the label for a specific window.

        Override this for supervised datasets. Default returns 0.
        """
        return 0

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec_idx, start = self._window_index[index]
        win_size = self.config.window_size_samples

        data = self._get_recording_data(rec_idx)
        window = data[:, start : start + win_size]

        # Per-channel z-normalization
        if self.config.normalize:
            mean = window.mean(axis=-1, keepdims=True)
            std = window.std(axis=-1, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            window = (window - mean) / std

            if self.config.clip_std > 0:
                window = np.clip(window, -self.config.clip_std, self.config.clip_std)

        label = self._get_label(rec_idx, start)

        return {
            "eeg": window.astype(np.float32),
            "label": label,
            "recording": self._recordings[rec_idx].stem,
            "window_start": start,
        }

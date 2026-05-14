from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import opensmile
except ImportError:  # pragma: no cover
    opensmile = None  # type: ignore[assignment]


def extract_egemaps(
    wav_path: str | Path,
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "LowLevelDescriptors",
) -> np.ndarray:
    """Return (T, 88) eGeMAPS frame-level features for a single WAV file."""
    if opensmile is None:
        raise ImportError("opensmile is required: pip install opensmile")

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set),
        feature_level=getattr(opensmile.FeatureLevel, feature_level),
    )
    df = smile.process_file(str(wav_path))
    return df.values.astype(np.float32)


def load_cached(cache_path: str | Path) -> np.ndarray:
    return np.load(str(cache_path)).astype(np.float32)


def cache_exists(recording_name: str, cache_dir: str | Path) -> bool:
    stem = Path(recording_name).stem
    return (Path(cache_dir) / f"{stem}.npy").exists()


def cache_path_for(recording_name: str, cache_dir: str | Path) -> Path:
    stem = Path(recording_name).stem
    return Path(cache_dir) / f"{stem}.npy"

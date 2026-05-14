"""Pre-extract eGeMAPS features for all recordings and cache as .npy files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.features import extract_egemaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract eGeMAPS features")
    parser.add_argument("--recordings_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--feature_set", default="eGeMAPSv02", help="opensmile FeatureSet name"
    )
    parser.add_argument(
        "--feature_level", default="LowLevelDescriptors", help="opensmile FeatureLevel name"
    )
    args = parser.parse_args()

    rec_dir = Path(args.recordings_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(rec_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files found in {rec_dir}")

    for wav in tqdm(wavs, desc="Extracting"):
        out_path = out_dir / f"{wav.stem}.npy"
        if out_path.exists():
            continue
        features = extract_egemaps(wav, args.feature_set, args.feature_level)
        np.save(out_path, features)

    print(f"Done. {len(wavs)} files → {out_dir}")


if __name__ == "__main__":
    main()

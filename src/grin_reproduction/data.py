"""AQI-36 preparation following the original GRIN out-of-sample protocol.

This module implements preprocessing independently. The methodological sources
are the pinned authors' ``air_quality.py``, ``utils.py`` and data-module code;
see PROTOCOL.md for links and documented compatibility changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATA_URL = "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/air/air_quality.zip"
ARCHIVE_SHA256 = "95d3c5cab14fc007f43c57b745a93e5f88201cbb92a755644ca9777848343317"
UPSTREAM_COMMIT = "4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e"


@dataclass
class PreparedData:
    values: np.ndarray
    raw_values: np.ndarray
    observed_mask: np.ndarray
    eval_mask: np.ndarray
    input_mask: np.ndarray
    adjacency: np.ndarray
    train_starts: np.ndarray
    val_starts: np.ndarray
    test_starts: np.ndarray
    mean: float
    std: float
    timestamps: pd.DatetimeIndex
    stations: pd.DataFrame
    provenance: dict
    window: int = 36


def download_aqi36(data_dir: str | Path) -> Path:
    """Fetch the versioned archive, verify it, and extract only AQI-36."""
    folder = Path(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "air_quality.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "GRIN-course-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    digest = sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Dataset archive checksum mismatch: {digest}; remove {archive} and retry.")
    # No unrestricted extractall: the archive is external input.
    target = folder / "small36.h5"
    with zipfile.ZipFile(archive) as zipped:
        content = zipped.read("small36.h5")
    if not target.exists() or sha256(target.read_bytes()).digest() != sha256(content).digest():
        target.write_bytes(content)
    return target


def _calendar_fill(frame: pd.DataFrame) -> pd.DataFrame:
    """Authors' weekly/hourly fallback for NaNs (always masked from loss)."""
    filled = frame.copy()
    groups = [
        [frame.index.year, frame.index.isocalendar().week, frame.index.hour],
        [frame.index.year, frame.index.month, frame.index.hour],
        [frame.index.month, frame.index.hour],
        [frame.index.hour],
    ]
    for group in groups:
        if not filled.isna().any().any():
            break
        filled = filled.fillna(filled.groupby(group).transform("mean"))
    return filled.ffill().bfill()


def _covered(starts: np.ndarray, window: int, n_steps: int) -> np.ndarray:
    covered = np.zeros(n_steps, dtype=bool)
    if len(starts):
        covered[(starts[:, None] + np.arange(window)).ravel()] = True
    return covered


def split_aqi36(timestamps: pd.DatetimeIndex, window: int = 36,
                val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce AirQuality.splitter, including window gaps and validation placement.

    Test windows begin AND end in March, June, September or December.
    Validation windows are placed immediately before each test month, with a
    window-length offset, exactly as in the released out-of-sample experiment.
    """
    if window < 1 or len(timestamps) < window:
        raise ValueError("Window must be positive and fit within the data.")
    starts = np.arange(len(timestamps) - window + 1, dtype=np.int64)
    month = np.asarray(timestamps.month)
    test_months = np.array([3, 6, 9, 12])
    begins_test = np.isin(month[starts], test_months)
    ends_test = np.isin(month[starts + window - 1], test_months)
    test = starts[begins_test & ends_test]
    candidates = starts[~begins_test & ~ends_test]
    val_per_month = int(val_fraction * len(candidates)) // len(test_months)
    if not len(test) or val_per_month < 1:
        raise ValueError("Calendar needs test months and enough validation windows.")
    delta = np.diff(test)
    first = test[1:][np.flatnonzero(delta > delta.min())]
    if len(first) < len(test_months):
        first = np.insert(first, 0, test[0])
    validation = np.concatenate([
        (np.arange(start - val_per_month, start) - window) % len(starts)
        for start in first
    ]).astype(np.int64)
    val_coverage = _covered(validation, window, len(timestamps))
    overlap = val_coverage[candidates[:, None] + np.arange(window)].any(axis=1)
    train = candidates[~overlap]
    for left, right in [(train, validation), (train, test), (validation, test)]:
        if np.any(_covered(left, window, len(timestamps)) & _covered(right, window, len(timestamps))):
            raise AssertionError("Temporal splits overlap at the observation level.")
    return train, validation, test


def prepare_aqi36(data_dir: str | Path, window: int = 36,
                  val_fraction: float = 0.1, adj_threshold: float = 0.1) -> PreparedData:
    """Load fixed masks, split, build the geographical graph and fit train-only scale."""
    source = download_aqi36(data_dir)
    frame = pd.DataFrame(pd.read_hdf(source, "pm25"))
    stations = pd.DataFrame(pd.read_hdf(source, "stations"))
    evaluation = pd.DataFrame(pd.read_hdf(source, "eval_mask"))
    if not frame.index.equals(evaluation.index) or not frame.columns.equals(evaluation.columns):
        raise ValueError("The supplied evaluation mask is not aligned with the readings.")
    # The released small36.h5 starts at 01:00, so contains 8,759 rows;
    # the paper rounds the duration to one year / 8,760 hours.
    if frame.shape != (8759, 36):
        raise ValueError(f"Unexpected AQI-36 shape {frame.shape}.")
    timestamps = pd.DatetimeIndex(frame.index)
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("AQI-36 timestamps must be unique and ordered.")
    raw = frame.to_numpy(dtype=np.float32)
    observed = np.isfinite(raw)
    eval_mask = evaluation.to_numpy(dtype=bool)
    if np.any(eval_mask & ~observed):
        raise ValueError("An evaluation target has no observed ground truth.")
    values = _calendar_fill(frame).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("AQI-36 preprocessing left nonfinite values.")
    train, validation, test = split_aqi36(timestamps, window, val_fraction)
    input_mask = observed & ~eval_mask
    train_coverage = _covered(train, window, len(frame))
    # Upstream fits its scaler BEFORE restoring held-out evaluation values in
    # training intervals. Match that ordering rather than silently changing it.
    scaler_values = values[train_coverage]
    scaler_mask = input_mask[train_coverage]
    masked = np.where(scaler_mask, scaler_values, np.nan)
    mean = float(np.nanmean(masked))
    std = float(np.nanstd(masked))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training scale must be finite and positive.")
    input_mask[train_coverage] = observed[train_coverage]
    # Haversine distance with the same mean Earth radius used by the authors.
    coordinates = np.radians(stations.loc[:, ["latitude", "longitude"]].to_numpy(dtype=float))
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    angular = (np.sin((latitude[:, None] - latitude[None, :]) / 2) ** 2
               + np.cos(latitude[:, None]) * np.cos(latitude[None, :])
               * np.sin((longitude[:, None] - longitude[None, :]) / 2) ** 2)
    distances = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(angular, 0, 1)))
    theta = distances.std()
    adjacency = np.exp(-np.square(distances / theta))
    adjacency[adjacency < adj_threshold] = 0
    np.fill_diagonal(adjacency, 0)
    provenance = {
        "dataset": "AQI-36", "url": DATA_URL, "archive_sha256": ARCHIVE_SHA256,
        "small36_sha256": sha256(source.read_bytes()).hexdigest(),
        "protocol_commit": UPSTREAM_COMMIT, "window": window, "stride": 1,
        "test_months": [3, 6, 9, 12], "val_fraction": val_fraction,
        "adj_threshold": adj_threshold, "graph_theta_km": float(theta),
        "mean": mean, "std": std,
        "train_windows": len(train), "val_windows": len(validation), "test_windows": len(test),
        "original_missing": int((~observed).sum()), "fixed_eval_targets": int(eval_mask.sum()),
        "date_start": str(timestamps[0]), "date_end": str(timestamps[-1]),
    }
    return PreparedData(values, raw, observed, eval_mask, input_mask,
                        adjacency.astype(np.float32), train, validation, test,
                        mean, std, timestamps, stations, provenance, window)


# A short alias keeps the notebook readable.
load_aqi36 = prepare_aqi36

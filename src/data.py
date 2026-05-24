from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import AppConfig
from .preprocessing import ParsedSignal, augment_signal, denoise_signal, extract_active_segments, read_signal_file, standardize_segment
from .utils import safe_name


@dataclass(frozen=True)
class Sample:
    signal: np.ndarray
    label_index: int
    label_name: str
    sample_id: str
    source_file: str
    sample_rate_hz: int


@dataclass(frozen=True)
class RunDataset:
    task_name: str
    task_title: str
    labels: list[str]
    samples: list[Sample]
    train_indices: list[int]
    test_indices: list[int]
    inventory: pd.DataFrame
    preview_signal: ParsedSignal | None


class EMGDataset(Dataset):
    def __init__(self, samples: list[Sample], augment: bool) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        signal_data = augment_signal(sample.signal) if self.augment else sample.signal
        return (
            torch.tensor(signal_data.copy(), dtype=torch.float32).unsqueeze(0),
            torch.tensor(sample.label_index, dtype=torch.long),
            sample.sample_id,
        )


def discover_signal_files(data_dir: Path, pattern: str) -> list[Path]:
    if not data_dir.exists():
        raise RuntimeError(f"Data directory does not exist: {data_dir}")
    return sorted(path for path in data_dir.rglob(pattern) if path.is_file())


def build_dataset(project_dir: Path, config: AppConfig) -> RunDataset:
    files = discover_signal_files(project_dir / config.data.data_dir, config.data.file_glob)
    if not files:
        raise RuntimeError("No signal files found. Check data.data_dir and data.file_glob in config.yaml.")

    label_lookup = {safe_name(label): index for index, label in enumerate(config.task.labels)}
    label_map = {safe_name(key): safe_name(value) for key, value in config.task.label_map.items()}
    samples: list[Sample] = []
    inventory_rows: list[dict] = []
    preview_signal: ParsedSignal | None = None

    for path in files:
        parsed = read_signal_file(path, config.data)
        if preview_signal is None and len(parsed.signal) > 0:
            preview_signal = parsed
        source_key = _resolve_source_key(path, config.task.label_source)
        mapped_key = label_map.get(source_key, source_key)
        status = "ignored"
        segments_found = 0

        if mapped_key in label_lookup and len(parsed.signal) > 0:
            denoised = denoise_signal(parsed.signal, parsed.sample_rate_hz, config.preprocessing)
            segments = extract_active_segments(denoised, parsed.sample_rate_hz, config.preprocessing)
            segments_found = len(segments)
            if segments:
                status = "used"
                label_index = label_lookup[mapped_key]
                label_name = config.task.labels[label_index]
                stem = safe_name(path.stem)
                relative_path = str(path.relative_to(project_dir).as_posix())
                for segment_index, segment in enumerate(segments):
                    samples.append(
                        Sample(
                            signal=standardize_segment(segment),
                            label_index=label_index,
                            label_name=label_name,
                            sample_id=f"{stem}_{segment_index:04d}",
                            source_file=relative_path,
                            sample_rate_hz=parsed.sample_rate_hz,
                        )
                    )
            else:
                status = "empty_after_segmentation"
        elif mapped_key in label_lookup:
            status = "empty_signal"

        inventory_rows.append(
            {
                "file_path": str(path.relative_to(project_dir).as_posix()),
                "source_key": source_key,
                "mapped_label": mapped_key if mapped_key in label_lookup else "",
                "selected_for_task": int(mapped_key in label_lookup),
                "status": status,
                "sample_rate_hz": parsed.sample_rate_hz,
                "raw_signal_length": int(len(parsed.signal)),
                "segment_count": int(segments_found),
            }
        )

    if not samples:
        raise RuntimeError("No usable samples were extracted. Check labels, label_map, and parsing settings in config.yaml.")

    labels = np.asarray([sample.label_index for sample in samples])
    indices = np.arange(len(samples))
    stratify = labels if np.min(np.bincount(labels)) >= 2 else None
    train_indices, test_indices = train_test_split(
        indices,
        test_size=config.training.test_size,
        random_state=config.training.seed,
        stratify=stratify,
    )

    return RunDataset(
        task_name=config.task.name,
        task_title=config.task.title,
        labels=list(config.task.labels),
        samples=samples,
        train_indices=train_indices.tolist(),
        test_indices=test_indices.tolist(),
        inventory=pd.DataFrame(inventory_rows),
        preview_signal=preview_signal,
    )


def _resolve_source_key(path: Path, label_source: str) -> str:
    if label_source == "stem":
        return safe_name(path.stem)
    if label_source == "parent":
        return safe_name(path.parent.name)
    raise RuntimeError(f"Unsupported task.label_source: {label_source}")

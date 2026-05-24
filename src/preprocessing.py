from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .config import DataConfig, PreprocessingConfig


@dataclass(frozen=True)
class ParsedSignal:
    file_path: Path
    header_lines: list[str]
    signal: np.ndarray
    sample_rate_hz: int


def read_signal_file(path: Path, config: DataConfig) -> ParsedSignal:
    lines: list[str] | None = None
    for encoding in config.encodings:
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise RuntimeError(f"Unable to decode file: {path}")

    header = lines[: config.header_lines]
    sample_rate = _extract_sample_rate(header, config.sample_rate_regex)
    values = _extract_signal_values(lines[config.header_lines :], config.value_column, config.delimiter_regex)
    return ParsedSignal(file_path=path, header_lines=header, signal=values, sample_rate_hz=sample_rate)


def _extract_sample_rate(header_lines: list[str], pattern: str) -> int:
    header_text = "\n".join(header_lines)
    match = re.search(pattern, header_text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError("Sample rate not found in file header. Update data.sample_rate_regex in config.yaml.")
    return int(match.group(1))


def _extract_signal_values(lines: list[str], value_column: int, delimiter_regex: str) -> np.ndarray:
    values: list[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [item for item in re.split(delimiter_regex, stripped) if item]
        if len(parts) <= value_column:
            continue
        try:
            values.append(float(parts[value_column]))
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float32)


def denoise_signal(signal_data: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    if len(signal_data) < max(100, fs // 5):
        return signal_data.astype(np.float32)

    notch_hz = config.notch_hz
    high_hz = min(config.bandpass_high_hz, fs * 0.45)
    low_hz = min(config.bandpass_low_hz, max(1.0, high_hz - 1.0))
    if low_hz >= high_hz:
        return signal_data.astype(np.float32)

    b_notch, a_notch = signal.iirnotch(notch_hz, config.notch_q, fs)
    notched = signal.filtfilt(b_notch, a_notch, signal_data)
    b_band, a_band = signal.butter(4, [low_hz / (0.5 * fs), high_hz / (0.5 * fs)], btype="band")
    return signal.filtfilt(b_band, a_band, notched).astype(np.float32)


def extract_active_segments(signal_data: np.ndarray, fs: int, config: PreprocessingConfig) -> list[np.ndarray]:
    segment_length = max(8, int(round(fs * config.segment_ms / 1000.0)))
    window_size = max(4, int(round(fs * config.window_ms / 1000.0)))
    if len(signal_data) < segment_length or len(signal_data) < window_size:
        return []

    rms = np.sqrt(np.convolve(signal_data**2, np.ones(window_size) / window_size, mode="valid"))
    threshold = float(np.mean(rms) + config.threshold_factor * np.std(rms))
    active_indices = np.where(rms > threshold)[0]
    if len(active_indices) == 0:
        return []

    split_indices = np.where(np.diff(active_indices) > window_size)[0] + 1
    groups = np.split(active_indices, split_indices)
    half = segment_length // 2
    segments: list[np.ndarray] = []
    for group in groups:
        if len(group) == 0:
            continue
        start, end = int(group[0]), int(group[-1] + window_size)
        peak_idx = start + int(np.argmax(rms[start:end]))
        seg_start = max(0, peak_idx - half)
        seg_end = seg_start + segment_length
        if seg_end <= len(signal_data):
            segments.append(signal_data[seg_start:seg_end].astype(np.float32))
    return segments


def standardize_segment(segment: np.ndarray) -> np.ndarray:
    mean = float(segment.mean())
    std = float(segment.std())
    if std < 1e-6:
        return (segment - mean).astype(np.float32)
    return ((segment - mean) / std).astype(np.float32)


def augment_signal(segment: np.ndarray) -> np.ndarray:
    augmented = segment.copy()
    augmented *= np.random.uniform(0.9, 1.1)
    augmented = np.roll(augmented, np.random.randint(-20, 21))
    augmented += np.random.normal(0.0, np.random.uniform(0.01, 0.05), size=augmented.shape).astype(np.float32)
    mask_length = np.random.randint(max(5, len(augmented) // 40), max(6, len(augmented) // 12))
    mask_start = np.random.randint(0, max(1, len(augmented) - mask_length))
    augmented[mask_start : mask_start + mask_length] *= np.random.uniform(0.3, 0.8)
    return augmented.astype(np.float32)

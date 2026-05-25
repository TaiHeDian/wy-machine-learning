from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskConfig:
    name: str
    title: str
    labels: list[str]
    label_source: str
    label_map: dict[str, str]
    confusion_color: str


@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    file_glob: str
    encodings: list[str]
    header_lines: int
    sample_rate_regex: str
    value_column: int
    delimiter_regex: str


@dataclass(frozen=True)
class PreprocessingConfig:
    segment_ms: int
    window_ms: int
    threshold_factor: float
    notch_hz: float
    notch_q: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    augment: bool


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    test_size: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    scheduler_factor: float
    scheduler_patience: int
    num_workers: int
    device: str
    enable_early_stopping: bool


@dataclass(frozen=True)
class ModelConfig:
    channels: list[int]
    kernel_sizes: list[int]
    pool_sizes: list[int]
    adaptive_pool_size: int
    embedding_dim: int
    dropout: float
    use_batch_norm: bool


@dataclass(frozen=True)
class OutputConfig:
    base_dir: str
    timestamp_format: str
    dpi: int
    image_formats: list[str]


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    run: Path
    figures: Path
    tables: Path
    model: Path
    report: Path
    config_snapshot: Path


@dataclass(frozen=True)
class AppConfig:
    task: TaskConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: ModelConfig
    output: OutputConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "task": {
        "name": "words_demo",
        "title": "Word Gesture Classification",
        "labels": ["no", "thanks", "come", "arrive", "drink", "like", "delight"],
        "label_source": "stem",
        "label_map": {},
        "confusion_color": "#2B6CB0",
    },
    "data": {
        "data_dir": "data",
        "file_glob": "*.txt",
        "encodings": ["utf-8", "gbk", "latin1"],
        "header_lines": 3,
        "sample_rate_regex": r"(\d+)\s*Hz",
        "value_column": 2,
        "delimiter_regex": r"[\s,\t]+",
    },
    "preprocessing": {
        "segment_ms": 1000,
        "window_ms": 100,
        "threshold_factor": 2.0,
        "notch_hz": 50.0,
        "notch_q": 30.0,
        "bandpass_low_hz": 20.0,
        "bandpass_high_hz": 400.0,
        "augment": True,
    },
    "training": {
        "seed": 42,
        "test_size": 0.2,
        "batch_size": 32,
        "epochs": 80,
        "learning_rate": 0.0005,
        "weight_decay": 0.0005,
        "patience": 20,
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "num_workers": 0,
        "device": "auto",
        "enable_early_stopping": True,
    },
    "model": {
        "channels": [32, 64, 128, 128],
        "kernel_sizes": [7, 5, 5, 3],
        "pool_sizes": [2, 2, 2, 1],
        "adaptive_pool_size": 16,
        "embedding_dim": 128,
        "dropout": 0.3,
        "use_batch_norm": True,
    },
    "output": {
        "base_dir": "out",
        "timestamp_format": "%Y%m%d_%H%M%S",
        "dpi": 600,
        "image_formats": ["png", "tif"],
    },
}


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(project_dir: Path) -> AppConfig:
    config_data = DEFAULT_CONFIG
    config_path = project_dir / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
        config_data = _merge_dict(DEFAULT_CONFIG, user_config)

    task = TaskConfig(
        name=str(config_data["task"]["name"]),
        title=str(config_data["task"]["title"]),
        labels=list(config_data["task"]["labels"]),
        label_source=str(config_data["task"]["label_source"]),
        label_map={str(key): str(value) for key, value in dict(config_data["task"]["label_map"]).items()},
        confusion_color=str(config_data["task"]["confusion_color"]),
    )
    data = DataConfig(
        data_dir=str(config_data["data"]["data_dir"]),
        file_glob=str(config_data["data"]["file_glob"]),
        encodings=list(config_data["data"]["encodings"]),
        header_lines=int(config_data["data"]["header_lines"]),
        sample_rate_regex=str(config_data["data"]["sample_rate_regex"]),
        value_column=int(config_data["data"]["value_column"]) - 1,
        delimiter_regex=str(config_data["data"]["delimiter_regex"]),
    )
    preprocessing = PreprocessingConfig(
        segment_ms=int(config_data["preprocessing"]["segment_ms"]),
        window_ms=int(config_data["preprocessing"]["window_ms"]),
        threshold_factor=float(config_data["preprocessing"]["threshold_factor"]),
        notch_hz=float(config_data["preprocessing"]["notch_hz"]),
        notch_q=float(config_data["preprocessing"]["notch_q"]),
        bandpass_low_hz=float(config_data["preprocessing"]["bandpass_low_hz"]),
        bandpass_high_hz=float(config_data["preprocessing"]["bandpass_high_hz"]),
        augment=bool(config_data["preprocessing"]["augment"]),
    )
    training = TrainingConfig(
        seed=int(config_data["training"]["seed"]),
        test_size=float(config_data["training"]["test_size"]),
        batch_size=int(config_data["training"]["batch_size"]),
        epochs=int(config_data["training"]["epochs"]),
        learning_rate=float(config_data["training"]["learning_rate"]),
        weight_decay=float(config_data["training"]["weight_decay"]),
        patience=int(config_data["training"]["patience"]),
        scheduler_factor=float(config_data["training"]["scheduler_factor"]),
        scheduler_patience=int(config_data["training"]["scheduler_patience"]),
        num_workers=int(config_data["training"]["num_workers"]),
        device=str(config_data["training"]["device"]),
        enable_early_stopping=bool(config_data["training"]["enable_early_stopping"]),
    )
    model = ModelConfig(
        channels=[int(x) for x in config_data["model"]["channels"]],
        kernel_sizes=[int(x) for x in config_data["model"]["kernel_sizes"]],
        pool_sizes=[int(x) for x in config_data["model"]["pool_sizes"]],
        adaptive_pool_size=int(config_data["model"]["adaptive_pool_size"]),
        embedding_dim=int(config_data["model"]["embedding_dim"]),
        dropout=float(config_data["model"]["dropout"]),
        use_batch_norm=bool(config_data["model"]["use_batch_norm"]),
    )
    output = OutputConfig(
        base_dir=str(config_data["output"]["base_dir"]),
        timestamp_format=str(config_data["output"]["timestamp_format"]),
        dpi=int(config_data["output"]["dpi"]),
        image_formats=[str(x) for x in config_data["output"]["image_formats"]],
    )
    return AppConfig(task=task, data=data, preprocessing=preprocessing, training=training, model=model, output=output)


def build_paths(project_dir: Path, config: AppConfig) -> Paths:
    run_name = datetime.now().strftime(config.output.timestamp_format)
    run_dir = project_dir / config.output.base_dir / run_name
    paths = Paths(
        root=project_dir,
        data=project_dir / config.data.data_dir,
        run=run_dir,
        figures=run_dir / "figures",
        tables=run_dir / "tables",
        model=run_dir / "model.pth",
        report=run_dir / "report.md",
        config_snapshot=run_dir / "config_resolved.yaml",
    )
    for path in [paths.run, paths.figures, paths.tables]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dump_config(config: AppConfig) -> dict[str, Any]:
    return asdict(config)


def save_config_snapshot(config: AppConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(dump_config(config), sort_keys=False, allow_unicode=True), encoding="utf-8")

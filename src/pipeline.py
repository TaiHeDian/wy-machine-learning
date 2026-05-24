from __future__ import annotations

from pathlib import Path

from .config import build_paths, load_config, save_config_snapshot
from .data import build_dataset
from .plotting import build_tsne_table, plot_confusion_matrix, plot_preprocessing, plot_training_curves, plot_tsne
from .preprocessing import denoise_signal
from .reporting import export_tables, write_report
from .training import train_model
from .utils import choose_device, set_seed


def run_pipeline(project_dir: Path | None = None) -> int:
    project_dir = project_dir or Path(__file__).resolve().parents[2]
    config = load_config(project_dir)
    paths = build_paths(project_dir, config)
    save_config_snapshot(config, paths.config_snapshot)
    set_seed(config.training.seed)
    device = choose_device(config.training.device)

    dataset = build_dataset(project_dir, config)
    result = train_model(dataset, config, device, paths.model)
    tsne_table = build_tsne_table(result)
    export_tables(result, dataset.inventory, tsne_table, paths)
    write_report(paths, config, result, dataset.inventory)

    if dataset.preview_signal is not None and len(dataset.preview_signal.signal) > 0:
        denoised = denoise_signal(dataset.preview_signal.signal, dataset.preview_signal.sample_rate_hz, config.preprocessing)
        plot_preprocessing(dataset.preview_signal.signal, denoised, dataset.preview_signal.sample_rate_hz, paths.figures, config)

    plot_confusion_matrix(result, paths.figures, config)
    plot_tsne(tsne_table, result, paths.figures, config)
    plot_training_curves(result, paths.figures, config)

    print(f"Using device: {device}")
    print(f"Task: {result.task_name}")
    print(f"Data directory: {paths.data}")
    print(f"Accuracy: {result.accuracy * 100:.2f}%")
    print(f"Macro-F1: {result.macro_f1:.4f}")
    print(f"Run output: {paths.run}")
    return 0

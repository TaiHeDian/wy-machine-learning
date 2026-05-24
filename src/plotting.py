from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import AppConfig
from .training import TaskResult


PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1", "#EDC948", "#8CD17D"]


def setup_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    latin = "Arial" if "Arial" in available else "DejaVu Sans"
    cjk = next((name for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if name in available), latin)
    plt.rcParams.update(
        {
            "font.family": [latin, cjk],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )
    sns.set_theme(style="white", context="paper", rc={"font.family": [latin, cjk]})


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, config: AppConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in config.output.image_formats:
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            bbox_inches="tight",
            transparent=False,
            facecolor="white",
            dpi=config.output.dpi,
        )
    plt.close(fig)


def plot_preprocessing(raw_signal: np.ndarray, denoised_signal: np.ndarray, sample_rate_hz: int, out_dir: Path, config: AppConfig) -> None:
    setup_style()
    time_axis = np.arange(len(raw_signal)) / sample_rate_hz
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    axes[0].plot(time_axis, raw_signal, color=PALETTE[0], linewidth=1.3)
    axes[0].set_title("Original Signal", weight="bold")
    axes[1].plot(time_axis, denoised_signal, color=PALETTE[3], linewidth=1.3)
    axes[1].set_title("Denoised Signal", weight="bold")
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Voltage (mV)")
        axis.grid(True, color="#E6E6E6", linewidth=0.7)
    save_figure(fig, out_dir, "preprocessing_overview", config)


def plot_training_curves(result: TaskResult, out_dir: Path, config: AppConfig) -> None:
    setup_style()
    history = result.history
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_acc"] * 100, label="Train Accuracy", color=PALETTE[0], linewidth=1.8)
    axes[0].plot(history["epoch"], history["test_acc"] * 100, label="Validation Accuracy", color=PALETTE[3], linewidth=1.8)
    axes[1].plot(history["epoch"], history["train_loss"], label="Train Loss", color=PALETTE[0], linewidth=1.8)
    axes[1].plot(history["epoch"], history["test_loss"], label="Validation Loss", color=PALETTE[3], linewidth=1.8)
    axes[0].set_title(f"{result.task_title} Accuracy", weight="bold")
    axes[1].set_title(f"{result.task_title} Loss", weight="bold")
    for axis, ylabel in zip(axes, ["Accuracy (%)", "Loss"]):
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#E6E6E6", linewidth=0.7)
        axis.legend(frameon=False)
    save_figure(fig, out_dir, "training_curves", config)


def build_tsne_table(result: TaskResult) -> pd.DataFrame:
    features = StandardScaler().fit_transform(np.nan_to_num(result.embeddings, nan=0.0, posinf=0.0, neginf=0.0))
    sample_count = features.shape[0]
    perplexity = min(30, max(5, sample_count // 4))
    perplexity = min(perplexity, max(1, sample_count - 1))
    if sample_count <= 3:
        coords = PCA(n_components=2, random_state=42).fit_transform(features)
    else:
        coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(features)

    rows = []
    for sample_id, label_index, (x_coord, y_coord) in zip(result.embedding_ids, result.embedding_labels, coords):
        rows.append(
            {
                "sample_id": sample_id,
                "label": result.labels[label_index],
                "x": float(x_coord),
                "y": float(y_coord),
                "perplexity": int(perplexity),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(result: TaskResult, out_dir: Path, config: AppConfig) -> None:
    setup_style()
    size = max(5.5, 0.42 * len(result.labels) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    divider = make_axes_locatable(ax)
    # 改为右侧 colorbar
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cmap = sns.light_palette(config.task.confusion_color, as_cmap=True)
    annot = np.array(
        [
            [
                f"{value:.0f}%"
                if abs(value - round(value)) < 0.05
                else f"{value:.1f}%"
                for value in row
            ]
            for row in result.confusion_percent
        ]
    )

    sns.heatmap(
        result.confusion_percent,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={
            "orientation": "vertical",
            "label": "Normalized percentage (%)",
        },
    )

    ax.set_title(
        f"{result.task_title}\nAccuracy = {result.accuracy * 100:.1f}%",
        fontsize=12,
        pad=10,
        weight="bold",
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticklabels(result.labels, rotation=0, ha="center")
    ax.set_yticklabels(result.labels, rotation=0)

    save_figure(fig, out_dir, "confusion_matrix", config)


def plot_tsne(tsne_table: pd.DataFrame, result: TaskResult, out_dir: Path, config: AppConfig) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for index, label in enumerate(result.labels):
        label_df = tsne_table[tsne_table["label"] == label]
        ax.scatter(
            label_df["x"],
            label_df["y"],
            s=28,
            color=PALETTE[index % len(PALETTE)],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.45,
            label=label,
        )
    ax.set_title(f"{result.task_title}\nt-SNE", fontsize=12, weight="bold", pad=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_figure(fig, out_dir, "tsne", config)

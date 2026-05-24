from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AppConfig, Paths
from .training import TaskResult
from .utils import write_text


def export_tables(result: TaskResult, inventory: pd.DataFrame, tsne_table: pd.DataFrame, paths: Paths) -> None:
    pd.DataFrame(
        [
            {
                "task_name": result.task_name,
                "task_title": result.task_title,
                "accuracy_percent": round(result.accuracy * 100, 3),
                "macro_f1": round(result.macro_f1, 6),
                "num_classes": len(result.labels),
                "num_test_samples": len(result.predictions),
                "model_path": paths.model.name,
            }
        ]
    ).to_csv(paths.tables / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    result.predictions.to_csv(paths.tables / "predictions.csv", index=False, encoding="utf-8-sig")
    result.history.to_csv(paths.tables / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result.confusion_counts, index=result.labels, columns=result.labels).to_csv(
        paths.tables / "confusion_counts.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(result.confusion_percent, index=result.labels, columns=result.labels).round(3).to_csv(
        paths.tables / "confusion_percent.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "label": label,
                "support": int(result.confusion_counts[index].sum()),
                "correct": int(result.confusion_counts.diagonal()[index]),
                "accuracy_percent": round(
                    (100.0 * result.confusion_counts.diagonal()[index] / result.confusion_counts[index].sum())
                    if result.confusion_counts[index].sum()
                    else 0.0,
                    3,
                ),
            }
            for index, label in enumerate(result.labels)
        ]
    ).to_csv(paths.tables / "per_class_accuracy.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(paths.tables / "signal_inventory.csv", index=False, encoding="utf-8-sig")
    tsne_table.to_csv(paths.tables / "tsne_embeddings.csv", index=False, encoding="utf-8-sig")


def write_report(paths: Paths, config: AppConfig, result: TaskResult, inventory: pd.DataFrame) -> None:
    skipped = inventory[(inventory["selected_for_task"] == 1) & (inventory["status"] != "used")]
    lines = [
        "# EMG Classification Report",
        "",
        f"- Task: `{result.task_name}`",
        f"- Title: {result.task_title}",
        f"- Data directory: `{config.data.data_dir}`",
        f"- Output directory: `{paths.run.relative_to(paths.root).as_posix()}`",
        f"- Accuracy: {result.accuracy * 100:.2f}%",
        f"- Macro-F1: {result.macro_f1:.4f}",
        "",
        "## Labels",
        "",
        ", ".join(result.labels),
        "",
        "## Output Files",
        "",
        "- `figures/preprocessing_overview.(png|tif)`",
        "- `figures/confusion_matrix.(png|tif)`",
        "- `figures/training_curves.(png|tif)`",
        "- `figures/tsne.(png|tif)`",
        "- `tables/metrics_summary.csv`",
        "- `tables/predictions.csv`",
        "- `tables/training_history.csv`",
        "- `tables/confusion_counts.csv`",
        "- `tables/confusion_percent.csv`",
        "- `tables/per_class_accuracy.csv`",
        "- `tables/signal_inventory.csv`",
        "- `tables/tsne_embeddings.csv`",
        f"- `{paths.model.name}`",
        f"- `{paths.config_snapshot.name}`",
        "",
        "## Data Notes",
        "",
    ]
    if skipped.empty:
        lines.append("- All selected files produced usable active segments.")
    else:
        lines.extend(f"- {row['file_path']} -> {row['status']}" for _, row in skipped.iterrows())
    write_text(paths.report, "\n".join(lines) + "\n")

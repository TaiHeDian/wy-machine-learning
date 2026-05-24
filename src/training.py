from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import AppConfig
from .data import EMGDataset, RunDataset
from .models import ConfigurableCNN


@dataclass(frozen=True)
class TaskResult:
    task_name: str
    task_title: str
    labels: list[str]
    accuracy: float
    macro_f1: float
    confusion_counts: np.ndarray
    confusion_percent: np.ndarray
    predictions: pd.DataFrame
    history: pd.DataFrame
    embeddings: np.ndarray
    embedding_labels: list[int]
    embedding_ids: list[str]
    model_path: Path


def train_model(dataset: RunDataset, config: AppConfig, device: torch.device, model_path: Path) -> TaskResult:
    model = ConfigurableCNN(num_classes=len(dataset.labels), config=config.model).to(device)
    train_loader = _make_loader(dataset.samples, dataset.train_indices, config.training.batch_size, config.preprocessing.augment, config.training.num_workers)
    test_loader = _make_loader(dataset.samples, dataset.test_indices, config.training.batch_size, False, config.training.num_workers)
    full_loader = DataLoader(EMGDataset(dataset.samples, augment=False), batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_eval: dict[str, object] | None = None
    best_acc = -1.0
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_stats = _train_epoch(model, train_loader, criterion, optimizer, device)
        eval_stats = evaluate_model(model, test_loader, criterion, device)
        scheduler.step(float(eval_stats["loss"]))

        row = {
            "epoch": epoch,
            "train_loss": round(float(train_stats["loss"]), 6),
            "train_acc": round(float(train_stats["accuracy"]), 6),
            "test_loss": round(float(eval_stats["loss"]), 6),
            "test_acc": round(float(eval_stats["accuracy"]), 6),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{config.training.epochs:03d} | "
            f"train_loss={row['train_loss']:.4f} | train_acc={row['train_acc'] * 100:.2f}% | "
            f"val_loss={row['test_loss']:.4f} | val_acc={row['test_acc'] * 100:.2f}% | "
            f"lr={row['lr']:.6f}"
        )

        if float(eval_stats["accuracy"]) > best_acc + 1e-8:
            best_acc = float(eval_stats["accuracy"])
            best_state = copy.deepcopy(model.state_dict())
            best_eval = eval_stats
            stale_epochs = 0
        else:
            stale_epochs += 1

        if config.training.enable_early_stopping and stale_epochs >= config.training.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    best_eval = best_eval or evaluate_model(model, test_loader, criterion, device)
    full_eval = evaluate_model(model, full_loader, criterion, device)

    confusion_counts = confusion_matrix(best_eval["y_true"], best_eval["y_pred"], labels=list(range(len(dataset.labels))))
    row_sum = confusion_counts.sum(axis=1, keepdims=True)
    confusion_percent = np.divide(
        confusion_counts * 100.0,
        row_sum,
        out=np.zeros_like(confusion_counts, dtype=np.float64),
        where=row_sum != 0,
    )
    predictions = pd.DataFrame(
        {
            "sample_id": best_eval["sample_ids"],
            "true_label": [dataset.labels[index] for index in best_eval["y_true"]],
            "predicted_label": [dataset.labels[index] for index in best_eval["y_pred"]],
            "is_correct": [int(true == pred) for true, pred in zip(best_eval["y_true"], best_eval["y_pred"])],
        }
    )

    return TaskResult(
        task_name=dataset.task_name,
        task_title=dataset.task_title,
        labels=dataset.labels,
        accuracy=float(best_eval["accuracy"]),
        macro_f1=float(best_eval["macro_f1"]),
        confusion_counts=confusion_counts,
        confusion_percent=confusion_percent,
        predictions=predictions,
        history=pd.DataFrame(history_rows),
        embeddings=np.asarray(full_eval["embeddings"]),
        embedding_labels=list(full_eval["y_true"]),
        embedding_ids=list(full_eval["sample_ids"]),
        model_path=model_path,
    )


def _make_loader(samples: list, indices: list[int], batch_size: int, augment: bool, num_workers: int) -> DataLoader:
    subset = [samples[index] for index in indices]
    return DataLoader(EMGDataset(subset, augment=augment), batch_size=batch_size, shuffle=augment, num_workers=num_workers)


def _train_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for inputs, labels, _ in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
    }


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[str] = []
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels, batch_ids in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, embedding = model(inputs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            sample_ids.extend(batch_ids)
            embeddings.append(embedding.cpu().numpy())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "sample_ids": sample_ids,
        "embeddings": np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0)),
    }

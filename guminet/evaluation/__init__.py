import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from guminet.utils import resolve_device, save_json


def classification_metrics(targets, predictions, class_names):
    labels = list(range(len(class_names)))
    report = classification_report(targets, predictions, labels=labels, output_dict=True, zero_division=0)
    return dict(
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=report["macro avg"]["f1-score"],
        weighted_f1=report["weighted avg"]["f1-score"],
        per_class={name: report[str(i)] for i, name in enumerate(class_names)},
        report=report,
        confusion_matrix=confusion_matrix(targets, predictions, labels=labels).tolist(),
    )


@torch.inference_mode()
def evaluate(model, loader, device="auto", class_names=None, output_dir=None):
    device = resolve_device(device)
    model.to(device)
    previous = model.training
    model.eval()
    targets, predictions, loss_sum, count = [], [], 0.0, 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            targets.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            count += y.numel()
    finally:
        model.train(previous)
    if not count:
        raise ValueError("Evaluation loader contains no samples")
    if class_names is None:
        class_names = loader.dataset.classes
    result = classification_metrics(targets, predictions, class_names)
    result["loss"] = loss_sum / count
    if output_dir:
        save_evaluation(result, class_names, output_dir)
    return result


def save_evaluation(result, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "evaluation.json", result)
    with (output_dir / "per_class.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for name in class_names:
            writer.writerow({"class": name, **result["per_class"][name]})
    from matplotlib.figure import Figure

    fig = Figure(figsize=(max(5, len(class_names) * 0.6), max(4, len(class_names) * 0.5)))
    ax = fig.subplots()
    matrix = np.array(result["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
    )
    for label in ax.get_xticklabels():
        label.set(rotation=45, ha="right")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")


def plot_history(history, output_dir):
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("loss", "accuracy"):
        fig = Figure()
        ax = fig.subplots()
        for prefix in ("train", "val"):
            ax.plot([h["epoch"] for h in history], [h[f"{prefix}_{metric}"] for h in history], label=prefix)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png")


def plot_comparison(rows, output_dir, metrics=("macro_f1", "cpu_latency_ms", "peak_process_ram_mb")):
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        selected = [row for row in rows if metric in row]
        if not selected:
            continue
        fig = Figure()
        ax = fig.subplots()
        ax.bar([r["name"] for r in selected], [r[metric] for r in selected])
        ax.set(ylabel=metric)
        for label in ax.get_xticklabels():
            label.set(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_{metric}.png")

"""Portable, content-based train/validation provenance. Never opens test for selection."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from guminet.models import model_architecture_summary
from guminet.utils import save_json


def source_identity():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    digest = hashlib.sha256()
    for path in sorted((root / "guminet").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(git_commit=commit, git_dirty=dirty, package_source_sha256=digest.hexdigest())


def split_manifest(dataset):
    digest = hashlib.sha256()
    if hasattr(dataset, "samples") and hasattr(dataset, "root"):
        for path, label in dataset.samples:
            file_digest = hashlib.sha256()
            with Path(path).open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            record = [Path(path).relative_to(dataset.root).as_posix(), int(label), file_digest.hexdigest()]
            digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
        method = "sha256 ordered relative paths, labels and full file contents"
    elif hasattr(dataset, "tensors"):
        for tensor in dataset.tensors:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((list(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
        method = "sha256 tensor shapes, dtypes and bytes (synthetic tests)"
    else:
        return dict(
            samples=len(dataset),
            sha256=None,
            method="unavailable for custom dataset; supply dataset_manifest_id",
        )
    return dict(samples=len(dataset), sha256=digest.hexdigest(), method=method)


def hardware_info():
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return dict(
        platform=platform.platform(),
        cpu=cpu,
        logical_cpus=psutil.cpu_count(),
        total_system_ram_mb=psutil.virtual_memory().total / 2**20,
        torch_threads=torch.get_num_threads(),
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    )


def create_experiment(model, config, train_loader, val_loader):
    names = ["torch", "numpy", "Pillow", "scikit-learn", "PyYAML", "psutil", "optuna", "onnx", "onnxruntime"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        schema_version=1,
        experiment_id=config.get("experiment_id") or uuid.uuid4().hex,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        **source_identity(),
        dataset_name=config.get("dataset_name", config.get("dataset", "custom_dataset")),
        dataset_manifest_id=config.get("dataset_manifest_id"),
        splits={"train": split_manifest(train_loader.dataset), "val": split_manifest(val_loader.dataset)},
        model_architecture=getattr(model, "architecture", "guminet"),
        model_config=model.model_config,
        num_classes=model.model_config["num_classes"],
        seed=config["seed"],
        training_config=config,
        dependency_versions=versions,
        hardware=hardware_info(),
        architecture_summary=model_architecture_summary(model, input_size=config["input_size"]),
    )


def save_run_report(output_dir, experiment, history, best_metric, stop_reason):
    output_dir = Path(output_dir)
    report = dict(
        schema_version=1,
        experiment=experiment,
        completed_epochs=len(history),
        stop_reason=stop_reason,
        best_validation_macro_f1=best_metric,
        validation_metrics=history[-1] if history else None,
        total_training_wall_seconds=sum(row["train_seconds"] for row in history),
        total_process_cpu_training_seconds=sum(row.get("process_cpu_train_seconds", 0) for row in history),
        timing_note="Training loop only; wall and process CPU seconds are distinct; excludes validation/checkpoint/plots",
        sampled_peak_training_ram_mb=max((r["sampled_peak_training_ram_mb"] for r in history), default=None),
        checkpoint_paths={n: str((output_dir / f"{n}.pt").resolve()) for n in ("best", "last")},
        model_selection_split="val",
    )
    save_json(output_dir / "experiment.json", experiment)
    save_json(output_dir / "run_report.json", report)
    flat = dict(
        experiment_id=experiment["experiment_id"],
        git_commit=experiment["git_commit"],
        source_sha256=experiment["package_source_sha256"],
        seed=experiment["seed"],
        parameters=experiment["architecture_summary"]["total_parameters"],
        estimated_macs=experiment["architecture_summary"]["estimated_macs"],
        val_macro_f1=best_metric,
        completed_epochs=len(history),
        stop_reason=stop_reason,
        total_training_wall_seconds=report["total_training_wall_seconds"],
        total_process_cpu_training_seconds=report["total_process_cpu_training_seconds"],
        sampled_peak_training_ram_mb=report["sampled_peak_training_ram_mb"],
    )
    with (output_dir / "run_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    lines = [
        "# Experiment run",
        "",
        *[f"- {k}: {v}" for k, v in flat.items()],
        "",
        "Validation-only model selection. Training wall time is not process CPU time.",
        "RSS sampling is not an exact transient peak; activation estimates are not RSS.",
        "Full source/config/dataset/hardware provenance: experiment.json and run_report.json.",
    ]
    (output_dir / "run_report.md").write_text("\n".join(lines) + "\n")
    return report

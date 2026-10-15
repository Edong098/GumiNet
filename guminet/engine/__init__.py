import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from guminet.data import MEAN, STD
from guminet.evaluation import evaluate, plot_history
from guminet.utils import MemoryMonitor, resolve_device, resource_metrics, save_json, seed_everything

from .checkpoint import atomic_checkpoint, load_checkpoint, model_from_checkpoint

PROFILES = {
    "low_resource": dict(input_size=128, batch_size=4, num_workers=0, device="auto", amp=False),
    "balanced": dict(input_size=160, batch_size=16, num_workers=0, device="auto", amp=False),
    "performance": dict(input_size=224, batch_size=32, num_workers=2, device="auto", amp=True),
}


class GumiTrainer:
    def __init__(self, model, config=None, class_to_idx=None, optimizer_groups=None):
        supplied = dict(config or {})
        profile = supplied.get("profile", "low_resource")
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        self.config = (
            dict(
                epochs=10,
                lr=1e-3,
                optimizer="adamw",
                weight_decay=1e-4,
                scheduler="cosine",
                accumulation_steps=1,
                clip_grad=None,
                patience=5,
                seed=42,
                output_dir="runs/default",
                progress=True,
                mean=list(MEAN),
                std=list(STD),
                **PROFILES[profile],
            )
            | supplied
        )
        c = self.config
        if (
            c["epochs"] < 1
            or c["accumulation_steps"] < 1
            or int(c["accumulation_steps"]) != c["accumulation_steps"]
        ):
            raise ValueError("epochs and accumulation_steps must be positive integers")
        seed_everything(c["seed"])
        self.device = resolve_device(c["device"])
        self.model = model.to(self.device)
        self.class_to_idx = dict(
            class_to_idx or {str(i): i for i in range(model.model_config["num_classes"])}
        )
        if len(self.class_to_idx) != model.model_config["num_classes"]:
            raise ValueError("Class mapping does not match classifier")
        params = (
            optimizer_groups
            if optimizer_groups is not None
            else [p for p in model.parameters() if p.requires_grad]
        )
        if c["optimizer"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])
        elif c["optimizer"].lower() == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=c["lr"], momentum=0.9, weight_decay=c["weight_decay"])
        else:
            raise ValueError("optimizer must be adamw or sgd")
        if c["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=c["epochs"])
        elif c["scheduler"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, c["epochs"] // 3)
            )
        elif c["scheduler"] in (None, "none"):
            self.scheduler = None
        else:
            raise ValueError("scheduler must be cosine, step, or none")
        self.amp = bool(c["amp"] and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.output_dir = Path(c["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history, self.start_epoch, self.best_metric, self.bad_epochs = [], 0, -1.0, 0
        self.experiment = None

    def _payload(self, epoch, train_loader):
        return dict(
            format_version=1,
            architecture=getattr(self.model, "architecture", "guminet"),
            model_config=self.model.model_config,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            scaler_state=self.scaler.state_dict(),
            epoch=epoch,
            best_metric=self.best_metric,
            bad_epochs=self.bad_epochs,
            class_to_idx=self.class_to_idx,
            input_size=self.config["input_size"],
            mean=list(self.config["mean"]),
            std=list(self.config["std"]),
            config=self.config,
            history=self.history,
            torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),
            numpy_rng=[
                np.random.get_state()[0],
                np.random.get_state()[1].tolist(),
                *np.random.get_state()[2:],
            ],
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            loader_rng=train_loader.generator.get_state() if train_loader.generator else None,
            experiment=self.experiment,
        )

    def resume(self, path):
        ckpt = load_checkpoint(path)
        if ckpt["model_config"] != self.model.model_config or ckpt["class_to_idx"] != self.class_to_idx:
            raise ValueError("Resume architecture/class mapping mismatch")
        for key in ("input_size", "mean", "std", "optimizer", "scheduler", "accumulation_steps"):
            if ckpt["config"].get(key) != self.config.get(key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt["scheduler_state"]:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric, self.bad_epochs = ckpt["best_metric"], ckpt["bad_epochs"]
        self.history = ckpt["history"]
        self.experiment = ckpt.get("experiment")
        self._resume_path = str(path)
        torch.set_rng_state(ckpt["torch_rng"])
        random.setstate(ckpt["python_rng"])
        state = ckpt["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        if self.device.type == "cuda" and ckpt["cuda_rng"]:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        self._loader_rng = ckpt["loader_rng"]
        return self

    def fit(self, train_loader, val_loader, epoch_callback=None):
        if not len(train_loader) or not len(val_loader):
            raise ValueError("Training and validation loaders must be nonempty")
        for loader in (train_loader, val_loader):
            if hasattr(loader.dataset, "class_to_idx") and loader.dataset.class_to_idx != self.class_to_idx:
                raise ValueError("Loader class mapping mismatch")
            transform = getattr(loader.dataset, "transform", None)
            if transform is not None and hasattr(transform, "size"):
                if (
                    transform.size != self.config["input_size"]
                    or tuple(transform.mean) != tuple(self.config["mean"])
                    or tuple(transform.std) != tuple(self.config["std"])
                ):
                    raise ValueError("Loader preprocessing does not match training configuration")
        if getattr(self, "_loader_rng", None) is not None and train_loader.generator:
            train_loader.generator.set_state(self._loader_rng)
        c = self.config
        from guminet.research.tracking import create_experiment, save_run_report, source_identity

        current = create_experiment(self.model, c, train_loader, val_loader)
        if self.experiment is None:
            self.experiment = current
            if getattr(self, "_resume_path", None):
                self.experiment["legacy_resume_checkpoint"] = self._resume_path
                self.experiment["provenance_scope"] = (
                    "Resume session only; original epoch provenance unavailable"
                )
                self.experiment["legacy_completed_epochs"] = self.start_epoch
        else:
            # A known split mismatch is never silently accepted on resume.
            for split in ("train", "val"):
                before = self.experiment["splits"][split]["sha256"]
                after = current["splits"][split]["sha256"]
                if before is not None and after != before:
                    raise ValueError(f"Resume dataset manifest mismatch: {split}")
            self.experiment.setdefault("resume_events", []).append(
                dict(timestamp_utc=current["timestamp_utc"], source=source_identity(), config=c)
            )
        save_json(self.output_dir / "experiment.json", self.experiment)
        stop_reason = "epoch_limit"
        time_budget = c.get("training_time_budget_seconds")
        if time_budget is not None and time_budget <= 0:
            raise ValueError("training_time_budget_seconds must be positive")
        budget_metric = c.get("training_time_budget_metric", "wall")
        if budget_metric not in ("wall", "process_cpu"):
            raise ValueError("training_time_budget_metric must be wall or process_cpu")
        budget_field = "train_seconds" if budget_metric == "wall" else "process_cpu_train_seconds"
        class_names = sorted(self.class_to_idx, key=self.class_to_idx.get)
        for epoch in range(self.start_epoch, c["epochs"]):
            started = time.perf_counter()
            cpu_started = time.process_time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.model.train()
            # Freeze BN running stats in explicitly frozen fine-tuning stages.
            for module in getattr(self.model, "features", self.model).modules():
                if isinstance(module, torch.nn.BatchNorm2d) and not any(
                    p.requires_grad for p in module.parameters()
                ):
                    module.eval()
            total_loss, correct, count = 0.0, 0, 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{c['epochs']}", disable=not c["progress"])
            self.optimizer.zero_grad(set_to_none=True)
            window_samples = 0
            with MemoryMonitor() as memory:
                for batch_idx, (x, y) in enumerate(progress):
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.autocast(device_type=self.device.type, enabled=self.amp):
                        logits = self.model(x)
                        loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
                    self.scaler.scale(loss).backward()
                    window_samples += len(y)
                    if (batch_idx + 1) % c["accumulation_steps"] == 0 or batch_idx + 1 == len(train_loader):
                        self.scaler.unscale_(self.optimizer)
                        # Normalize by actual samples, including a short final accumulation window.
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(window_samples)
                        if c["clip_grad"] is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["clip_grad"])
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        window_samples = 0
                    total_loss += loss.item()
                    correct += (logits.argmax(1) == y).sum().item()
                    count += len(y)
                    progress.set_postfix(loss=loss.item() / len(y), accuracy=correct / count)
            train_seconds = time.perf_counter() - started
            cpu_train_seconds = time.process_time() - cpu_started
            val = evaluate(self.model, val_loader, self.device, class_names)
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()
            improved = val["macro_f1"] > self.best_metric
            if improved:
                self.best_metric, self.bad_epochs = val["macro_f1"], 0
            else:
                self.bad_epochs += 1
            row = dict(
                epoch=epoch + 1,
                train_loss=total_loss / count,
                train_accuracy=correct / count,
                val_loss=val["loss"],
                val_accuracy=val["accuracy"],
                val_macro_f1=val["macro_f1"],
                learning_rate=lr,
                epoch_seconds=time.perf_counter() - started,
                train_seconds=train_seconds,
                process_cpu_train_seconds=cpu_train_seconds,
                images_per_second=count / train_seconds,
                sampled_peak_training_ram_mb=memory.peak_mb,
                **resource_metrics(self.device),
            )
            self.history.append(row)
            save_json(self.output_dir / "history.json", self.history)
            with (self.output_dir / "history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.history)
            payload = self._payload(epoch, train_loader)
            atomic_checkpoint(self.output_dir / "last.pt", payload)
            if improved:
                atomic_checkpoint(self.output_dir / "best.pt", payload)
            if c["progress"]:
                tqdm.write(
                    f"val loss={val['loss']:.4f}, accuracy={val['accuracy']:.3f}, "
                    f"F1={val['macro_f1']:.3f}, lr={lr:.3g}, "
                    f"{row['images_per_second']:.1f} images/s, RAM={row['ram_mb']:.1f} MB, "
                    f"VRAM peak={row['cuda_peak_mb']:.1f} MB, epoch={row['epoch_seconds']:.2f}s"
                )
            save_run_report(self.output_dir, self.experiment, self.history, self.best_metric, stop_reason)
            if epoch_callback:
                epoch_callback(epoch, row)
            # Equal CPU wall-time budgets are checked at epoch boundaries; disclose overshoot.
            if time_budget is not None and sum(h.get(budget_field, 0) for h in self.history) >= time_budget:
                stop_reason = f"training_{budget_metric}_time_budget_epoch_boundary"
                break
            if c["patience"] and self.bad_epochs >= c["patience"]:
                stop_reason = "early_stopping"
                break
        if self.history:
            plot_history(self.history, self.output_dir)
        save_run_report(self.output_dir, self.experiment, self.history, self.best_metric, stop_reason)
        return self.history


def prepare_finetuning(checkpoint, num_classes, mode="head", unfreeze_last=2, backbone_lr=1e-4, head_lr=1e-3):
    model, metadata = model_from_checkpoint(checkpoint)
    if metadata.get("architecture", "guminet") != "guminet":
        raise ValueError("Fine-tuning helper currently supports GumiNet checkpoints only")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.model_config["num_classes"] = num_classes
    if mode not in ("head", "partial", "full"):
        raise ValueError("mode must be head, partial, or full")
    if mode == "partial" and not 1 <= unfreeze_last <= len(model.features):
        raise ValueError("unfreeze_last must select existing feature stages")
    for parameter in model.features.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "partial":
        for stage in list(model.features)[-unfreeze_last:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    groups = [{"params": model.classifier.parameters(), "lr": head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return model, groups

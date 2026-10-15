"""Measured timings; Conv/Linear MAC estimates are explicitly not complete FLOP accounting."""

import csv
import platform
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from guminet.models import count_parameters, create_model
from guminet.utils import MemoryMonitor, resolve_device, resource_metrics, save_json


def estimate_macs(model, example):
    macs, output_bytes, handles = 0, 0, []

    def hook(module, inputs, output):
        nonlocal macs, output_bytes
        if isinstance(output, torch.Tensor):
            output_bytes += output.numel() * output.element_size()
        if isinstance(module, nn.Conv2d):
            macs += (
                output.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    previous = model.training
    try:
        for module in model.modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(hook))
        model.eval()
        with torch.no_grad():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    return dict(
        estimated_macs=macs, estimated_conv_linear_flops=2 * macs, summed_leaf_output_bytes=output_bytes
    )


class GumiBenchmark:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = resolve_device(device)

    def run(
        self,
        input_size=160,
        batch_size=1,
        warmup=3,
        repeats=10,
        train_steps=3,
        output_dir=None,
        checkpoint=None,
    ):
        if repeats < 1 or warmup < 0 or train_steps < 1 or batch_size < 1:
            raise ValueError("Positive repeats/train_steps/batch_size and nonnegative warmup required")
        # Work on a copy so benchmarking never updates the caller's weights or BN buffers.
        import copy

        model = copy.deepcopy(self.model).to(self.device)
        x = torch.randn(batch_size, 3, input_size, input_size, device=self.device)
        y = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        metrics = estimate_macs(model, x)
        model.eval()

        def sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        latency = []
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            sync()
            for _ in range(repeats):
                start = time.perf_counter()
                model(x)
                sync()
                latency.append((time.perf_counter() - start) * 1000)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        model.train()
        forward, backward, step = [], [], []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with MemoryMonitor() as memory:
            for i in range(warmup + train_steps):
                optimizer.zero_grad(set_to_none=True)
                sync()
                start = time.perf_counter()
                loss = nn.functional.cross_entropy(model(x), y)
                sync()
                end_forward = time.perf_counter()
                loss.backward()
                sync()
                end_backward = time.perf_counter()
                optimizer.step()
                sync()
                end_step = time.perf_counter()
                if i >= warmup:
                    forward.append(end_forward - start)
                    backward.append(end_backward - end_forward)
                    step.append(end_step - end_backward)
        duration = sum(forward) + sum(backward) + sum(step)
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), state_file)
            state_bytes = state_file.stat().st_size
        import psutil

        cpu_model = platform.processor() or platform.machine()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        metrics.update(
            cpu_model=cpu_model,
            logical_cpu_count=psutil.cpu_count(),
            total_system_ram_mb=psutil.virtual_memory().total / 2**20,
            parameters=count_parameters(model),
            trainable_parameters=count_parameters(model, True),
            fp32_weight_bytes=4 * count_parameters(model),
            serialized_state_dict_bytes=state_bytes,
            checkpoint_file_bytes=Path(checkpoint).stat().st_size if checkpoint else None,
            inference_batch_latency_ms=statistics.median(latency),
            inference_p95_batch_ms=float(__import__("numpy").percentile(latency, 95)),
            forward_ms=1000 * statistics.mean(forward),
            backward_ms=1000 * statistics.mean(backward),
            optimizer_step_ms=1000 * statistics.mean(step),
            synthetic_train_duration_seconds=duration,
            synthetic_train_images_per_second=batch_size * train_steps / duration,
            input_size=input_size,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
            device=str(self.device),
            torch_version=str(torch.__version__),
            python=platform.python_version(),
            platform=platform.platform(),
            cpu=platform.processor() or platform.machine(),
            cpu_threads=torch.get_num_threads(),
            sampled_peak_training_ram_mb=memory.peak_mb,
            **resource_metrics(self.device),
        )
        metrics["cpu_latency_ms" if self.device.type == "cpu" else "gpu_latency_ms"] = statistics.median(
            latency
        )
        if self.device.type == "cuda":
            metrics["gpu"] = torch.cuda.get_device_name(self.device)
        if output_dir:
            save_json(Path(output_dir) / "benchmark.json", metrics)
            with (Path(output_dir) / "benchmark.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics))
                writer.writeheader()
                writer.writerow(metrics)
        return metrics


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.classifier(self.layers(x).flatten(1))


def _create_comparison_model(name, num_classes, pretrained=False):
    if name.startswith("guminet"):
        return create_model(name, num_classes, pretrained)
    if name == "simple_cnn":
        if pretrained:
            raise ValueError("SimpleCNN has no pretrained weights")
        return SimpleCNN(num_classes)
    supported = {"mobilenet_v2", "mobilenet_v3_small", "shufflenet_v2_x0_5", "efficientnet_v2_s"}
    if name in supported:
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("Install guminet[benchmark] for torchvision comparisons") from exc
        builder = getattr(models, name)
        model = builder(weights="DEFAULT" if pretrained else None)
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name in ("ghostnet_100", "mobilenetv4_conv_small"):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm for optional GhostNet/MobileNetV4 adapters") from exc
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported comparison model: {name}")


def create_comparison_model(name, num_classes, pretrained=False):
    model = _create_comparison_model(name, num_classes, pretrained)
    if not name.startswith("guminet"):
        model.architecture = "comparison"
        model.model_config = dict(name=name, num_classes=num_classes)
    return model

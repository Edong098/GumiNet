"""Sequential fresh-process CPU measurements. No fork-inherited high-water marks."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from guminet.utils import save_json


def profile_isolated(config):
    root = Path(config.get("output_dir", "runs/profile")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    models = config.get("models", [dict(name="guminet_nano"), dict(name="guminet_tiny")])
    for index, entry in enumerate(models):
        entry = dict(name=entry) if isinstance(entry, str) else dict(entry)
        settings = {k: v for k, v in config.items() if k not in ("models", "output_dir")}
        settings.update(entry)
        settings.setdefault("input_size", 128)
        output = root / f"model_{index:02}.json"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite profiling result: {output}")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(settings))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "guminet.benchmark.isolated",
                    "--worker",
                    str(config_path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise RuntimeError(f"Profiler worker {entry['name']} failed: {completed.stderr}")
        results.append(json.loads(output.read_text()))
    save_json(
        root / "profile.json", dict(process_isolation="fresh exec per model, sequential", results=results)
    )
    import csv

    flat = [
        dict(
            name=r["name"],
            experiment_id=r["experiment_id"],
            timestamp_utc=r["timestamp_utc"],
            process_id=r["process_id"],
            **r["metrics"],
        )
        for r in results
    ]
    fields = list(dict.fromkeys(k for r in flat for k in r))
    with (root / "profile.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    (root / "profile.md").write_text(
        "# Isolated CPU profiling\n\n"
        + "\n".join(
            f"- {r['name']}: {r['metrics']['parameters']} parameters; {r['metrics']['cpu_latency_ms']:.3f} ms/batch; "
            f"process {r['process_id']}"
            for r in results
        )
        + "\n\nWarm-up/repeats and hardware are recorded in JSON. RSS sampling is not exact peak.\n"
    )
    return results


def worker(config, output):
    import os
    import uuid
    from datetime import datetime, timezone

    import torch

    from guminet.benchmark import GumiBenchmark
    from guminet.models import create_model, model_architecture_summary
    from guminet.research.tracking import source_identity
    from guminet.utils import seed_everything

    torch.set_num_threads(config.get("threads", 2))
    seed_everything(config.get("seed", 42))
    model = create_model(
        config["name"], num_classes=config.get("num_classes", 3), **config.get("model_kwargs", {})
    )
    metrics = GumiBenchmark(model, "cpu").run(
        **{
            k: config[k]
            for k in ("input_size", "batch_size", "warmup", "repeats", "train_steps")
            if k in config
        }
    )
    metrics["memory_scope"] = "whole fresh worker, including original model plus benchmark copy and runtime"
    # Profile after timings to keep diagnostic overhead out of the timing loop.
    summary = model_architecture_summary(
        model,
        input_size=config.get("input_size", 128),
        batch_size=config.get("batch_size", 1),
        include_gradients=True,
    )
    save_json(
        output,
        dict(
            experiment_id=uuid.uuid4().hex,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            source=source_identity(),
            config=config,
            data_source="Random synthetic tensors; no classification dataset or test-set selection",
            name=config["name"],
            model_config=model.model_config,
            process_id=os.getpid(),
            metrics=metrics,
            architecture_summary=summary,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worker(json.loads(Path(args.worker).read_text()), args.output)

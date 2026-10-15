"""Train comparison adapters with the same engine, split, seed, and epoch budget."""

import csv
from pathlib import Path

from guminet.benchmark import GumiBenchmark, create_comparison_model
from guminet.data import make_loaders
from guminet.engine import PROFILES, GumiTrainer
from guminet.engine.checkpoint import model_from_checkpoint
from guminet.evaluation import evaluate, plot_comparison
from guminet.utils import save_json, seed_everything


def run_comparisons(config):
    root = Path(config.get("output_dir", "runs/comparison"))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in config.get("models", ["guminet_tiny", "simple_cnn", "mobilenet_v2"]):
        c = PROFILES[config.get("profile", "low_resource")] | config | dict(output_dir=str(root / name))
        seed_everything(c.get("seed", 42))
        loaders = make_loaders(
            c["dataset"],
            c["input_size"],
            c["batch_size"],
            c["num_workers"],
            c.get("seed", 42),
            **{k: c[k] for k in ("mean", "std") if k in c},
        )
        mapping = loaders["train"].dataset.class_to_idx
        pretrained = name in config.get("pretrained_models", [])
        model = create_comparison_model(name, len(mapping), pretrained)
        trainer = GumiTrainer(model, c, mapping)
        history = trainer.fit(loaders["train"], loaders["val"])
        checkpoint = root / name / "best.pt"
        best, _ = model_from_checkpoint(checkpoint)
        split = "test" if "test" in loaders else "val"
        metrics = evaluate(
            best, loaders[split], c.get("device", "auto"), output_dir=root / name / "evaluation"
        )
        bench = GumiBenchmark(best, "cpu").run(
            c["input_size"], warmup=1, repeats=c.get("repeats", 10), checkpoint=checkpoint
        )
        rows.append(
            dict(
                name=name,
                pretrained=pretrained,
                evaluation_split=split,
                accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"],
                **bench,
                epoch_seconds=history[-1]["epoch_seconds"],
                dataset_train_images_per_second=history[-1]["images_per_second"],
                dataset_sampled_peak_training_ram_mb=history[-1]["sampled_peak_training_ram_mb"],
            )
        )
        save_json(root / "comparison.json", dict(config=config, results=rows))
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(
        rows,
        root,
        metrics=(
            "accuracy",
            "macro_f1",
            "cpu_latency_ms",
            "dataset_sampled_peak_training_ram_mb",
            "epoch_seconds",
        ),
    )
    return rows

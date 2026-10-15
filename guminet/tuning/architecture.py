"""One-factor-at-a-time ablations. Measurements are observations, never superiority claims."""

import copy
import csv
from pathlib import Path

from guminet.benchmark import GumiBenchmark
from guminet.data import make_loaders
from guminet.engine import GumiTrainer
from guminet.engine.checkpoint import model_from_checkpoint
from guminet.evaluation import evaluate, plot_comparison
from guminet.models import create_model
from guminet.utils import save_json, seed_everything


def ablation_configs():
    base = dict(
        model="guminet_v2_plain",
        model_kwargs=dict(depth=2, width=1.0, expansion=1.0, residual=False),
        input_size=160,
    )
    changes = {
        "width": dict(model_kwargs=dict(width=0.75)),
        "depth": dict(model_kwargs=dict(depth=1)),
        "downsampling": dict(model_kwargs=dict(strides=[1, 2, 1, 2, 1, 2])),
        "expansion": dict(model_kwargs=dict(expansion=1.5)),
        "residual": dict(model_kwargs=dict(residual=True)),
        "resolution": dict(input_size=128),
    }
    result = {"baseline": base}
    for name, change in changes.items():
        variant = copy.deepcopy(base)
        variant["model_kwargs"].update(change.get("model_kwargs", {}))
        if "input_size" in change:
            variant["input_size"] = change["input_size"]
        result[name] = variant
    return result


def run_ablations(config):
    root = Path(config.get("output_dir", "runs/ablations"))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, variant in ablation_configs().items():
        c = dict(config) | variant | dict(output_dir=str(root / name))
        seed_everything(c.get("seed", 42))
        loaders = make_loaders(
            c["dataset"],
            c["input_size"],
            c.get("batch_size", 4),
            seed=c.get("seed", 42),
            include_test=False,
            **{k: c[k] for k in ("mean", "std") if k in c},
        )
        mapping = loaders["train"].dataset.class_to_idx
        model = create_model(c["model"], len(mapping), **c["model_kwargs"])
        trainer = GumiTrainer(model, c, mapping)
        history = trainer.fit(loaders["train"], loaders["val"])
        best, _ = model_from_checkpoint(root / name / "best.pt")
        val = evaluate(best, loaders["val"], "cpu")
        bench = GumiBenchmark(best).run(
            c["input_size"],
            warmup=1,
            repeats=c.get("repeats", 5),
            train_steps=2,
            checkpoint=root / name / "best.pt",
        )
        row = dict(
            name=name,
            accuracy=val["accuracy"],
            macro_f1=val["macro_f1"],
            **bench,
            training_images_per_second=history[-1]["images_per_second"],
            dataset_sampled_peak_training_ram_mb=history[-1]["sampled_peak_training_ram_mb"],
            epoch_seconds=history[-1]["epoch_seconds"],
        )
        rows.append(row)
        save_json(root / "ablations.json", dict(experiments=ablation_configs(), results=rows))
    with (root / "ablations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows, root)
    return rows

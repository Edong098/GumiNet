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


def _legacy_ablation_configs():
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


def ablation_configs(group="legacy"):
    if group == "legacy":
        return _legacy_ablation_configs()
    from guminet.research.presets import experiment_presets

    return experiment_presets(group)


def run_ablations(config):
    root = Path(config.get("output_dir", "runs/ablations"))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    variants = config.get("experiments") or ablation_configs(config.get("group", "legacy"))
    if not isinstance(variants, dict) or not variants:
        raise ValueError("experiments must be a nonempty mapping")
    seeds = config.get("seeds", [config.get("seed", 42)])
    for name, variant in variants.items():
        for seed in seeds:
            c = (
                dict(config)
                | variant
                | variant.get("training_overrides", {})
                | dict(
                    seed=seed,
                    output_dir=str(root / name / f"seed_{seed}")
                    if ("seeds" in config or config.get("group", "legacy") != "legacy")
                    else str(root / name),
                )
            )
            if (Path(c["output_dir"]) / "last.pt").exists():
                raise FileExistsError(f"Run already exists: {c['output_dir']}; choose a new output_dir")
            seed_everything(c.get("seed", 42))
            loaders = make_loaders(
                c["dataset"],
                c["input_size"],
                c.get("batch_size", 4),
                seed=c.get("seed", 42),
                include_test=False,
                augmentation=c.get("augmentation", True),
                **{k: c[k] for k in ("mean", "std") if k in c},
            )
            mapping = loaders["train"].dataset.class_to_idx
            model = create_model(c["model"], len(mapping), **c["model_kwargs"])
            trainer = GumiTrainer(model, c, mapping)
            history = trainer.fit(loaders["train"], loaders["val"])
            best, _ = model_from_checkpoint(Path(c["output_dir"]) / "best.pt")
            val = evaluate(best, loaders["val"], "cpu")
            bench = GumiBenchmark(best).run(
                c["input_size"],
                warmup=1,
                repeats=c.get("repeats", 5),
                train_steps=2,
                checkpoint=Path(c["output_dir"]) / "best.pt",
            )
            row = dict(
                name=name,
                control=variant.get("control", "baseline"),
                seed=seed,
                accuracy=val["accuracy"],
                macro_f1=val["macro_f1"],
                **bench,
                training_images_per_second=history[-1]["images_per_second"],
                dataset_sampled_peak_training_ram_mb=history[-1]["sampled_peak_training_ram_mb"],
                epoch_seconds=history[-1]["epoch_seconds"],
            )
            rows.append(row)
            save_json(root / "ablations.json", dict(experiments=variants, results=rows))
    with (root / "ablations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows, root)
    return rows

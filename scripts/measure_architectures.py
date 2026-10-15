"""Regenerate untrained structural diagnostics; not classification evidence."""

import argparse

import torch

from guminet.models import MODEL_REGISTRY, GumiBlockV2, create_model, model_architecture_summary
from guminet.models.analysis import residual_path_probe
from guminet.research.presets import experiment_presets
from guminet.research.tracking import source_identity
from guminet.utils import save_json, seed_everything


def measure(output):
    torch.set_num_threads(2)
    reports = {}
    for name in MODEL_REGISTRY:
        seed_everything(42)
        model = create_model(name, 3)
        summary = model_architecture_summary(model, include_gradients=True)
        if name == "v2_candidate_stage3":
            summary["residual_branch_probes"] = [
                residual_path_probe(block, 48)
                for block in model.modules()
                if isinstance(block, GumiBlockV2) and block.use_residual
            ]
        reports[name] = dict(model_config=model.model_config, **summary)
    ablations = {}
    for group in ("v1", "v2"):
        ablations[group] = {}
        for name, config in experiment_presets(group).items():
            seed_everything(42)
            model = create_model(config["model"], 3, **config["model_kwargs"])
            ablations[group][name] = model_architecture_summary(model)
    save_json(
        output,
        dict(
            source="codex_local_architecture_analysis",
            source_identity=source_identity(),
            input_size=128,
            num_classes=3,
            model_initialization="Seed 42 before EVERY model; untrained random weights",
            classification_evidence=False,
            registry=reports,
            ablations=ablations,
        ),
    )
    print(
        f"Measured {len(reports)} registry models and {sum(map(len, ablations.values()))} ablation architectures: {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/architecture_measurements.json")
    args = parser.parse_args()
    measure(args.output)

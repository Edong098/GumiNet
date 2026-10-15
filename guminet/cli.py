import argparse
import json
import platform

import torch

from guminet import __version__
from guminet.data import make_loaders
from guminet.engine import PROFILES, GumiTrainer
from guminet.engine.checkpoint import model_from_checkpoint
from guminet.models import MODEL_REGISTRY, count_parameters, create_model
from guminet.utils import load_config, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description="GumiNet experimental CPU-first CNN toolkit")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads (e.g. 2 on a laptop)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show runtime and package versions")
    sub.add_parser("models", help="List models with measured parameter counts (3 classes)")
    for name in (
        "train",
        "evaluate",
        "tune",
        "benchmark",
        "export",
        "ablate",
        "compare",
        "profile",
        "architecture",
    ):
        p = sub.add_parser(name, help=f"Run {name} using YAML configuration")
        p.add_argument("--config", required=True, help="Path to YAML configuration")
    p = sub.add_parser("predict", help="Classify an image from a trained checkpoint")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.command == "info":
        result = dict(
            version=__version__,
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda_available=torch.cuda.is_available(),
            threads=torch.get_num_threads(),
        )
    elif args.command == "models":
        result = {name: count_parameters(create_model(name, num_classes=3)) for name in MODEL_REGISTRY}
    elif args.command == "predict":
        from guminet.inference import GumiPredictor

        result = GumiPredictor.from_checkpoint(args.checkpoint, args.device).predict(args.image)
    else:
        c = load_config(args.config)
        if "threads" in c:
            torch.set_num_threads(c["threads"])
        seed_everything(c.get("seed", 42))
        if args.command == "train":
            merged = PROFILES[c.get("profile", "low_resource")] | c
            loaders = make_loaders(
                c["dataset"],
                merged["input_size"],
                merged["batch_size"],
                merged["num_workers"],
                c.get("seed", 42),
                include_test=False,
                **{k: c[k] for k in ("mean", "std", "augmentation") if k in c},
            )
            mapping = loaders["train"].dataset.class_to_idx
            groups = None
            if "finetune_checkpoint" in c:
                from guminet.engine import prepare_finetuning

                model, groups = prepare_finetuning(
                    c["finetune_checkpoint"],
                    len(mapping),
                    mode=c.get("finetune_mode", "head"),
                    unfreeze_last=c.get("unfreeze_last", 2),
                    backbone_lr=c.get("backbone_lr", 1e-4),
                    head_lr=c.get("lr", 1e-3),
                )
            else:
                model = create_model(
                    c.get("model", "guminet_tiny"), len(mapping), **c.get("model_kwargs", {})
                )
            trainer = GumiTrainer(model, c, mapping, groups)
            if c.get("resume"):
                trainer.resume(c["resume"])
            result = trainer.fit(loaders["train"], loaders["val"])
        elif args.command == "evaluate":
            from torch.utils.data import DataLoader

            from guminet.data import FolderDataset, ImageTransform
            from guminet.evaluation import evaluate

            model, meta = model_from_checkpoint(c["checkpoint"])
            dataset = FolderDataset(
                c["dataset"],
                ImageTransform(meta["input_size"], False, meta["mean"], meta["std"]),
                meta["class_to_idx"],
            )
            result = evaluate(
                model,
                DataLoader(dataset, batch_size=c.get("batch_size", 4)),
                c.get("device", "auto"),
                output_dir=c.get("output_dir", "runs/evaluation"),
            )
        elif args.command == "tune":
            from guminet.tuning import GumiTuner

            study = GumiTuner(c).run()
            result = dict(best_params=study.best_params, best_macro_f1=study.best_value)
        elif args.command == "benchmark":
            from guminet.benchmark import GumiBenchmark, create_comparison_model

            model = (
                model_from_checkpoint(c["checkpoint"])[0]
                if c.get("checkpoint")
                else create_comparison_model(
                    c.get("model", "guminet_tiny"), c.get("num_classes", 3), c.get("pretrained", False)
                )
            )
            if c.get("model_kwargs") and not c.get("checkpoint"):
                model = create_model(
                    c.get("model", "guminet_tiny"), c.get("num_classes", 3), **c["model_kwargs"]
                )
            result = GumiBenchmark(model, c.get("device", "cpu")).run(
                **{
                    k: c[k]
                    for k in (
                        "input_size",
                        "batch_size",
                        "warmup",
                        "repeats",
                        "train_steps",
                        "output_dir",
                        "checkpoint",
                    )
                    if k in c
                }
            )
            result["pretrained"] = c.get("pretrained", False)
        elif args.command == "profile":
            from guminet.benchmark.isolated import profile_isolated

            result = profile_isolated(c)
        elif args.command == "architecture":
            from guminet.models import model_architecture_summary

            model = create_model(
                c.get("model", "guminet_tiny"), c.get("num_classes", 3), **c.get("model_kwargs", {})
            )
            result = model_architecture_summary(
                model,
                input_size=c.get("input_size", 128),
                include_gradients=c.get("include_gradients", False),
            )
        elif args.command == "compare":
            from guminet.benchmark.comparison import run_comparisons

            result = run_comparisons(c)
        elif args.command == "ablate":
            from guminet.tuning.architecture import run_ablations

            result = run_ablations(c)
        else:
            from guminet.export import export_onnx, quantize_onnx, validate_quantized

            result = export_onnx(c["checkpoint"], c["output"], validate=c.get("validate", True))
            if c.get("quantized_output"):
                quantize_onnx(c["output"], c["quantized_output"])
                result["quantization"] = validate_quantized(
                    c["output"], c["quantized_output"], result["input_size"]
                )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()

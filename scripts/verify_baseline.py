"""Reproducible synthetic end-to-end validation; never a real accuracy benchmark."""

import argparse
import platform
import shutil
import tempfile
from pathlib import Path

import torch
from make_synthetic import generate

from guminet import GumiPredictor, GumiTrainer, create_model
from guminet.benchmark import GumiBenchmark
from guminet.data import make_loaders
from guminet.engine.checkpoint import model_from_checkpoint
from guminet.evaluation import evaluate
from guminet.export import export_onnx, quantize_onnx, validate_quantized
from guminet.models import MODEL_REGISTRY, count_parameters
from guminet.utils import save_json, seed_everything


def verify(output):
    torch.set_num_threads(2)
    seed_everything(42)
    with tempfile.TemporaryDirectory(prefix="guminet-verification-") as directory:
        root = Path(directory)
        dataset = generate(root / "dataset")
        loaders = make_loaders(dataset, input_size=128, batch_size=4, seed=42)
        model = create_model("guminet_nano", 3)
        trainer = GumiTrainer(
            model,
            dict(
                epochs=10,
                input_size=128,
                seed=42,
                progress=False,
                accumulation_steps=2,
                scheduler="cosine",
                output_dir=str(root / "run"),
            ),
            loaders["train"].dataset.class_to_idx,
        )
        history = trainer.fit(loaders["train"], loaders["val"])
        checkpoint = root / "run/best.pt"
        best, metadata = model_from_checkpoint(checkpoint)
        metrics = evaluate(best, loaders["test"], "cpu", output_dir=root / "evaluation")
        predictor = GumiPredictor.from_checkpoint(checkpoint, "cpu")
        prediction = predictor.predict(dataset / "test/class_0/0.png")
        benchmark = GumiBenchmark(best).run(
            input_size=128, warmup=3, repeats=20, train_steps=5, checkpoint=checkpoint
        )
        exported = export_onnx(checkpoint, root / "model.onnx")
        quantize_onnx(root / "model.onnx", root / "classifier_int8.onnx")
        quantized = validate_quantized(root / "model.onnx", root / "classifier_int8.onnx", 128)
        summary = dict(
            purpose="Synthetic software validation only, not evidence of real-world accuracy",
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda_available=torch.cuda.is_available(),
            seed=42,
            dataset_samples={k: len(v.dataset) for k, v in loaders.items()},
            parameter_counts_3_classes={n: count_parameters(create_model(n, 3)) for n in MODEL_REGISTRY},
            epochs=len(history),
            first_train_loss=history[0]["train_loss"],
            last_train_loss=history[-1]["train_loss"],
            best_epoch=metadata["epoch"] + 1,
            synthetic_test_accuracy=metrics["accuracy"],
            synthetic_test_macro_f1=metrics["macro_f1"],
            last_epoch=history[-1],
            example_prediction=prediction,
            benchmark=benchmark,
            onnx=exported,
            classifier_quantization=quantized,
        )
        summary["onnx"]["path"] = "temporary validation artifact (not distributed)"
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "synthetic_validation.json", summary)
        # Retain only small reports/plots; never distribute synthetic checkpoints or data.
        for name in ("history.json", "history.csv", "loss.png", "accuracy.png"):
            shutil.copy2(root / "run" / name, output / name)
        for name in ("evaluation.json", "per_class.csv", "confusion_matrix.png"):
            shutil.copy2(root / "evaluation" / name, output / name)
        print(f"Validated CPU workflow; report: {output / 'synthetic_validation.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/validation/baseline")
    args = parser.parse_args()
    verify(args.output)

import io

import numpy as np
import pytest
import torch
from PIL import Image
from test_workflow import train_small

from guminet.benchmark import GumiBenchmark, create_comparison_model
from guminet.models import create_model
from guminet.tuning.architecture import ablation_configs


def test_benchmark(tmp_path):
    model = create_model("guminet_nano", 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = GumiBenchmark(model).run(input_size=32, warmup=1, repeats=2, train_steps=2, output_dir=tmp_path)
    assert result["parameters"] > 0 and result["estimated_macs"] > 0
    assert result["cpu_latency_ms"] > 0 and result["backward_ms"] > 0
    assert result["peak_process_ram_mb"] > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize(
    "name", ["simple_cnn", "mobilenet_v2", "mobilenet_v3_small", "shufflenet_v2_x0_5", "efficientnet_v2_s"]
)
def test_comparison(name):
    pytest.importorskip("torchvision")
    model = create_comparison_model(name, 3).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 3)


def test_ablations_one_factor():
    configs = ablation_configs()
    base = configs["baseline"]
    for name, config in configs.items():
        if name == "baseline":
            continue
        differences = sum(
            base["model_kwargs"].get(k) != config["model_kwargs"].get(k)
            for k in base["model_kwargs"].keys() | config["model_kwargs"].keys()
        )
        differences += config["input_size"] != base["input_size"]
        assert differences == 1


def test_tune(dataset_root, tmp_path):
    pytest.importorskip("optuna")
    from guminet.tuning import GumiTuner

    config = dict(
        dataset=str(dataset_root),
        output_dir=str(tmp_path / "tune"),
        trials=1,
        epochs_per_trial=1,
        input_sizes=[32],
        batch_sizes=[2],
        progress=False,
        threads=2,
    )
    study = GumiTuner(config).run()
    assert len(study.trials) == 1 and 0 <= study.best_value <= 1
    study2 = GumiTuner(config).run()
    assert len(study2.trials) == 2
    assert (tmp_path / "tune/best_params.json").is_file()


def test_onnx(dataset_root, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from guminet.export import export_onnx, quantize_onnx, validate_quantized

    trainer, loaders = train_small(dataset_root, tmp_path)
    trainer.fit(loaders["train"], loaders["val"])
    output = tmp_path / "export.onnx"
    result = export_onnx(tmp_path / "run/best.pt", output)
    assert result["validated"] and result["max_absolute_error"] < 1e-4
    quantize_onnx(output, tmp_path / "quantized.onnx")
    report = validate_quantized(output, tmp_path / "quantized.onnx", 32, repeats=2)
    assert np.isfinite(report["max_absolute_error"])


def test_api(dataset_root, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from guminet.inference.api import create_app

    trainer, loaders = train_small(dataset_root, tmp_path)
    trainer.fit(loaders["train"], loaders["val"])
    client = TestClient(create_app(tmp_path / "run/best.pt", "cpu", max_upload_bytes=4096))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model-info").json()["classes"] == ["a", "b"]
    payload = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(payload, format="PNG")
    assert (
        client.post("/predict", files={"file": ("a.png", payload.getvalue(), "image/png")}).status_code == 200
    )
    assert client.post("/predict", files={"file": ("a.txt", b"abc", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("a.png", b"bad", "image/png")}).status_code == 400
    assert client.post("/predict", files={"file": ("a.png", b"a" * 4097, "image/png")}).status_code == 413


def test_comparison_training(dataset_root, tmp_path):
    from guminet.benchmark.comparison import run_comparisons

    rows = run_comparisons(
        dict(
            dataset=str(dataset_root),
            output_dir=str(tmp_path / "comparison"),
            models=["guminet_nano", "simple_cnn"],
            input_size=32,
            batch_size=2,
            epochs=1,
            progress=False,
            repeats=2,
        )
    )
    assert len(rows) == 2
    assert all(r["evaluation_split"] == "test" for r in rows)
    assert all(r["dataset_sampled_peak_training_ram_mb"] > 0 for r in rows)


def test_ablation_training(dataset_root, tmp_path):
    from guminet.tuning.architecture import run_ablations

    rows = run_ablations(
        dict(
            dataset=str(dataset_root),
            output_dir=str(tmp_path / "ablations"),
            epochs=1,
            progress=False,
            batch_size=2,
            repeats=1,
        )
    )
    assert len(rows) == 7
    assert {row["name"] for row in rows} == set(ablation_configs())
    assert all(row["estimated_macs"] > 0 for row in rows)


@pytest.mark.parametrize("name", ["ghostnet_100", "mobilenetv4_conv_small"])
def test_timm_comparison(name):
    pytest.importorskip("timm")
    model = create_comparison_model(name, 3).eval()
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 3)

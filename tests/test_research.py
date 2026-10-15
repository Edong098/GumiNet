import json
import os
from pathlib import Path

import pytest
import torch
from test_workflow import train_small

from guminet.benchmark.isolated import profile_isolated
from guminet.models import create_model, model_architecture_summary
from guminet.research.presets import experiment_presets
from guminet.research.tracking import split_manifest
from guminet.tuning.architecture import run_ablations


def test_tracking_and_split_identity(dataset_root, tmp_path):
    trainer, loaders = train_small(dataset_root, tmp_path)
    trainer.fit(loaders["train"], loaders["val"])
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    exp = report["experiment"]
    assert exp["splits"]["train"]["sha256"] and set(exp["splits"]) == {"train", "val"}
    assert exp["package_source_sha256"] and exp["git_commit"]
    assert exp["architecture_summary"]["estimated_macs"] > 0
    assert report["total_process_cpu_training_seconds"] > 0
    from guminet.engine.checkpoint import load_checkpoint

    assert load_checkpoint(tmp_path / "run/last.pt")["experiment"]["experiment_id"] == exp["experiment_id"]
    assert (tmp_path / "run/run_report.csv").is_file() and (tmp_path / "run/run_report.md").is_file()
    original = split_manifest(loaders["train"].dataset)["sha256"]
    # Relative path/content manifest detects a changed image without opening the test split.
    path = dataset_root / "train/a/0.png"
    from PIL import Image

    Image.new("RGB", (32, 32), "green").save(path)
    assert split_manifest(loaders["train"].dataset)["sha256"] != original
    resumed, loaders = train_small(dataset_root, tmp_path, epochs=2)
    resumed.resume(tmp_path / "run/last.pt")
    with pytest.raises(ValueError, match="manifest mismatch"):
        resumed.fit(loaders["train"], loaders["val"])


@pytest.mark.parametrize("metric", ["wall", "process_cpu"])
def test_equal_time_budget(dataset_root, tmp_path, metric):
    trainer, loaders = train_small(
        dataset_root,
        tmp_path,
        epochs=5,
        training_time_budget_seconds=1e-9,
        training_time_budget_metric=metric,
    )
    history = trainer.fit(loaders["train"], loaders["val"])
    assert len(history) == 1
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["stop_reason"] == f"training_{metric}_time_budget_epoch_boundary"


def test_new_presets_cpu_and_controls():
    for group in ("v1", "v2"):
        configs = experiment_presets(group)
        for name, c in configs.items():
            model = create_model(c["model"], num_classes=3, **c["model_kwargs"])
            logits = model(torch.randn(2, 3, 32, 32))
            logits.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert name == "baseline" or c["control"] in configs
    configs = experiment_presets("v2")
    a = configs["candidate_residual"]["model_kwargs"].copy()
    b = configs["single_repeat_control"]["model_kwargs"].copy()
    assert a.pop("residual") and not b.pop("residual") and a == b
    assert model_architecture_summary(create_model("v2_candidate_stage3", 3))["total_parameters"] == 19975


def test_training_only_fixed_architecture():
    configs = experiment_presets("training")
    for c in configs.values():
        assert c["model"] == configs["baseline"]["model"]
        assert c["model_kwargs"] == configs["baseline"]["model_kwargs"]


def test_custom_yaml_ablation_no_test_access(dataset_root, tmp_path):
    # Unusable test folder must not be touched by a selection run.
    (dataset_root / "test/b/extra.txt").write_text("unused")
    experiments = {
        name: dict(model="v2_refined_d1_w075", model_kwargs=kwargs, input_size=32)
        for name, kwargs in [
            ("control", {}),
            ("candidate", dict(stage_depths=[1, 1, 1, 2, 1, 1], residual=True)),
        ]
    }
    rows = run_ablations(
        dict(
            dataset=str(dataset_root),
            output_dir=str(tmp_path / "ablate"),
            experiments=experiments,
            seeds=[42],
            batch_size=2,
            epochs=1,
            progress=False,
            repeats=1,
        )
    )
    assert len(rows) == 2 and all(r["seed"] == 42 for r in rows)
    with pytest.raises(FileExistsError):
        run_ablations(
            dict(
                dataset=str(dataset_root),
                output_dir=str(tmp_path / "ablate"),
                experiments=experiments,
                seeds=[42],
            )
        )


def test_fresh_process_profiler(tmp_path):
    rows = profile_isolated(
        dict(
            models=["guminet_nano", "v2_candidate_stage3"],
            input_size=32,
            repeats=1,
            warmup=1,
            train_steps=1,
            output_dir=str(tmp_path / "profile"),
        )
    )
    assert len({r["process_id"] for r in rows}) == 2
    assert len({r["experiment_id"] for r in rows}) == 2
    assert all(r["timestamp_utc"] and r["source"]["package_source_sha256"] for r in rows)
    assert all(r["process_id"] != os.getpid() for r in rows)
    assert all(r["metrics"]["cpu_latency_ms"] > 0 for r in rows)
    assert rows[1]["architecture_summary"]["active_residual_blocks"] == 1


def test_historical_data_integrity():
    data = json.loads(
        (Path(__file__).parents[1] / "experiments/historical/user_reported_colab.json").read_text()
    )
    assert not data["reproduced_by_codex"]
    assert len(data["multiseed_runs"]) == 15
    assert all(r["source"] == "user_reported_google_colab" for r in data["multiseed_runs"])
    for r in data["test_results"]:
        matrix = r["confusion_matrix"]
        assert sum(map(sum, matrix)) == 300
        assert sum(matrix[i][i] for i in range(3)) == r["correct"]


def test_legacy_missing_cpu_history_not_fabricated(tmp_path):
    from guminet.research.tracking import save_run_report

    experiment = dict(
        experiment_id="legacy-test",
        git_commit=None,
        package_source_sha256="unavailable",
        seed=42,
        architecture_summary=dict(total_parameters=1, estimated_macs=1),
    )
    history = [dict(train_seconds=2, sampled_peak_training_ram_mb=1)]
    report = save_run_report(tmp_path, experiment, history, 0.5, "epoch_limit")
    assert report["total_process_cpu_training_seconds"] is None
    assert not report["process_cpu_history_complete"]

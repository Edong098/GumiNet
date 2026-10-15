import json
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize(
    "command",
    ["info", "models", "train", "evaluate", "tune", "benchmark", "predict", "export", "ablate", "compare"],
)
def test_help(command):
    result = subprocess.run([sys.executable, "-m", "guminet.cli", command, "--help"], capture_output=True)
    assert result.returncode == 0 and b"usage:" in result.stdout


def test_cli_end_to_end(dataset_root, tmp_path):
    checkpoint = tmp_path / "run/best.pt"
    configs = {
        "train": dict(
            dataset=str(dataset_root),
            model="guminet_nano",
            input_size=32,
            batch_size=2,
            epochs=1,
            output_dir=str(tmp_path / "run"),
            progress=False,
            threads=2,
        ),
        "evaluate": dict(
            dataset=str(dataset_root / "test"),
            checkpoint=str(checkpoint),
            output_dir=str(tmp_path / "eval"),
            threads=2,
        ),
        "benchmark": dict(model="guminet_nano", input_size=32, repeats=2, warmup=1, train_steps=1, threads=2),
    }
    for command, config in configs.items():
        path = tmp_path / f"{command}.yaml"
        path.write_text(yaml.safe_dump(config))
        result = subprocess.run(
            [sys.executable, "-m", "guminet.cli", command, "--config", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guminet.cli",
            "--threads",
            "2",
            "predict",
            "--image",
            str(dataset_root / "test/a/0.png"),
            "--checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "confidence" in json.loads(result.stdout)

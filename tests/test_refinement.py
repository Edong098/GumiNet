import importlib.util
from pathlib import Path

import pytest
import torch

from guminet.engine.checkpoint import model_from_checkpoint
from guminet.models import GumiBlockV2, create_model, model_architecture_summary
from guminet.models.analysis import residual_path_probe

spec = importlib.util.spec_from_file_location(
    "guminet_legacy_reference", Path(__file__).parent / "fixtures/legacy_models_v010.py"
)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
OLD_NAMES = tuple(legacy.MODEL_REGISTRY)


@pytest.mark.parametrize("name", OLD_NAMES)
def test_legacy_structure_weights_and_config(name, tmp_path):
    torch.manual_seed(5)
    old = legacy.create_model(name, num_classes=3).eval()
    new = create_model(name, num_classes=3).eval()
    assert new.model_config == old.model_config
    assert list(new.state_dict()) == list(old.state_dict())
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)
    path = tmp_path / "legacy_generated_test_only.pt"
    torch.save(
        dict(
            format_version=1,
            architecture="guminet",
            model_config=old.model_config,
            model_state=old.state_dict(),
            class_to_idx={"a": 0, "b": 1, "c": 2},
            input_size=32,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        path,
    )
    restored, _ = model_from_checkpoint(path)
    restored.eval()
    torch.testing.assert_close(restored(x), old(x), rtol=0, atol=0)


def test_v1_v2_equivalence():
    v1 = create_model("guminet_v1", num_classes=3, width=0.75).eval()
    v2 = create_model("v2_refined_d1_w075", num_classes=3).eval()
    # Explicit ordered leaf mapping for a diagnostic, not a checkpoint conversion API.
    for a, b in zip(
        [m for m in v1.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear))],
        [m for m in v2.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear))],
        strict=True,
    ):
        b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    torch.testing.assert_close(v1(x), v2(y), rtol=0, atol=0)
    v1(x).sum().backward()
    v2(y).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert model_architecture_summary(v2)["active_residual_blocks"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_identity_gradient(enabled):
    block = GumiBlockV2(24, 24, residual=enabled)
    result = residual_path_probe(block)
    assert result["gradient_min"] == result["gradient_max"] == int(enabled)
    assert result["identity_output"] == enabled


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kernel_size=5),
        dict(activation="leaky_relu"),
        dict(strides=[1, 1, 2, 2, 2, 1]),
        dict(channels=[16, 24, 36, 48, 64, 80, 128]),
    ],
)
def test_opt_in_v1(kwargs):
    model = create_model("guminet_tiny", num_classes=3, **kwargs)
    logits = model(torch.randn(2, 3, 128, 128))
    logits.sum().backward()
    assert logits.shape == (2, 3)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("projection", ["post", "pre", "linear"])
def test_candidate_observability_roundtrip(projection, tmp_path):
    model = create_model("v2_candidate_stage3", num_classes=3, projection_activation=projection)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.get_rng_state().clone()
    report = model_architecture_summary(model, include_gradients=True)
    assert report["active_residual_blocks"] == 1
    assert 17000 <= report["total_parameters"] <= 30000
    assert all(b["finite_gradient"] for b in report["blocks"])
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    path = tmp_path / "candidate.pt"
    torch.save(
        dict(
            format_version=1,
            model_config=model.model_config,
            model_state=model.state_dict(),
            class_to_idx={"a": 0, "b": 1, "c": 2},
            input_size=128,
            mean=[0, 0, 0],
            std=[1, 1, 1],
        ),
        path,
    )
    loaded, _ = model_from_checkpoint(path)
    assert loaded.model_config == model.model_config
    assert model_architecture_summary(loaded)["active_residual_blocks"] == 1


@pytest.mark.parametrize("cin,cout,stride", [(16, 24, 1), (24, 24, 2)])
def test_incompatible_residual_forward(cin, cout, stride):
    block = GumiBlockV2(cin, cout, stride, residual=True)
    x = torch.randn(2, cin, 16, 16, requires_grad=True)
    y = block(x)
    assert y.shape == (2, cout, 16 // stride, 16 // stride) and not block.use_residual
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_baseline_shapes_and_historical_grid():
    report = model_architecture_summary(create_model("guminet_tiny", 3))
    assert [b["output_shape"][2] for b in report["blocks"]] == [64, 32, 32, 16, 16, 8, 8]
    for depth, width, expected in [
        (1, 0.5, 8407),
        (1, 0.75, 17047),
        (1, 1, 28907),
        (2, 0.5, 19355),
        (2, 0.75, 39675),
        (2, 1, 67603),
    ]:
        for residual in (False, True):
            model = create_model("guminet_v2_plain", 3, depth=depth, width=width, residual=residual)
            assert sum(p.numel() for p in model.parameters()) == expected


def test_summary_does_not_reseed_global_devices(monkeypatch):
    model = create_model("v2_candidate_stage3", 3)

    def forbid_global_reseed(*args, **kwargs):
        raise AssertionError("Summary must not reset global CPU/CUDA seeds")

    monkeypatch.setattr(torch, "manual_seed", forbid_global_reseed)
    report = model_architecture_summary(model, include_gradients=True)
    assert report["active_residual_blocks"] == 1

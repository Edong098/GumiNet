import pytest
import torch

from guminet.models import MODEL_REGISTRY, GumiBlockV1, GumiBlockV2, GumiNetV1, count_parameters, create_model


@pytest.mark.parametrize("name", MODEL_REGISTRY)
@pytest.mark.parametrize("classes,size", [(2, 32), (3, 128), (7, 160), (2, 192), (3, 224)])
def test_forward_backward(name, classes, size):
    model = create_model(name, num_classes=classes)
    output = model(torch.randn(2, 3, size, size))
    assert output.shape == (2, classes)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_baseline_and_family():
    model = GumiNetV1(num_classes=3)
    assert model.features[0][0].in_channels == 3
    assert model.features[0][0].out_channels == 16
    assert sum(isinstance(m, GumiBlockV1) for m in model.modules()) == 6
    assert model.classifier.in_features == 128
    counts = [count_parameters(create_model(n, 3)) for n in ("guminet_nano", "guminet_tiny", "guminet_small")]
    assert counts[0] < counts[1] < counts[2]
    assert len(create_model("guminet_small", 3).features) > len(model.features)


def test_residual_compatibility():
    block = GumiBlockV2(16, 16, residual=True)
    assert block.use_residual
    assert not GumiBlockV2(16, 24, residual=True).use_residual
    assert not GumiBlockV2(16, 16, stride=2, residual=True).use_residual
    for name in ("guminet_v2_residual", "guminet_v2_low_expansion_residual"):
        assert any(m.use_residual for m in create_model(name).modules() if isinstance(m, GumiBlockV2))


def test_invalid_model():
    with pytest.raises(ValueError, match="Pretrained"):
        create_model(pretrained=True)
    with pytest.raises(ValueError, match="Unknown"):
        create_model("missing")
    with pytest.raises(ValueError):
        GumiNetV1(num_classes=0)

"""Standard-operator research CNNs. No pretrained weights are distributed."""

import torch
from torch import nn

RESOLUTIONS = (128, 160, 192, 224)


def make_activation(name="relu"):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError("activation must be relu or leaky_relu")


def conv_bn_relu(cin, cout, kernel=3, stride=1, groups=1, activation="relu"):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        make_activation(activation),
    )


class GumiBlockV1(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, activation="relu"):
        super().__init__(
            conv_bn_relu(
                in_channels,
                in_channels,
                kernel=kernel_size,
                stride=stride,
                groups=in_channels,
                activation=activation,
            ),
            conv_bn_relu(in_channels, out_channels, kernel=1, activation=activation),
        )


class GumiBlockV2(nn.Module):
    """Spatial extraction, optional narrow mixing, linear projection, optional identity."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=1.0,
        residual=False,
        kernel_size=3,
        activation="relu",
        projection_activation="post",
    ):
        super().__init__()
        if expansion < 1:
            raise ValueError("expansion must be >= 1")
        hidden = max(in_channels, round(in_channels * expansion))
        layers = [
            conv_bn_relu(
                in_channels,
                in_channels,
                kernel=kernel_size,
                stride=stride,
                groups=in_channels,
                activation=activation,
            )
        ]
        if hidden != in_channels:
            layers.append(conv_bn_relu(in_channels, hidden, kernel=1, activation=activation))
        layers.extend([nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
        self.layers = nn.Sequential(*layers)
        self.use_residual = residual and stride == 1 and in_channels == out_channels
        if projection_activation not in ("post", "pre", "linear"):
            raise ValueError("projection_activation must be post, pre, or linear")
        self.projection_activation = projection_activation
        self.activation = make_activation(activation)

    def forward(self, x):
        y = self.layers(x)
        if self.projection_activation == "pre":
            y = self.activation(y)
        y = y + x if self.use_residual else y
        return self.activation(y) if self.projection_activation == "post" else y


class GumiNetV1(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        dropout=0.2,
        width=1.0,
        depth=1,
        channels=None,
        strides=None,
        version="v1",
        expansion=1.0,
        residual=False,
        kernel_size=3,
        activation="relu",
        stage_depths=None,
        residual_stages=None,
        projection_activation="post",
    ):
        super().__init__()
        if num_classes < 1 or width <= 0 or depth < 1 or int(depth) != depth:
            raise ValueError("Positive num_classes/width and integer depth >= 1 required")
        if version not in ("v1", "v2"):
            raise ValueError("version must be v1 or v2")
        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be 3 or 5")
        make_activation(activation)
        if projection_activation not in ("post", "pre", "linear"):
            raise ValueError("projection_activation must be post, pre, or linear")
        if version == "v1" and (residual_stages is not None or projection_activation != "post"):
            raise ValueError("residual_stages/projection_activation apply only to V2")
        depth = int(depth)
        base = channels or [16, 24, 32, 48, 64, 96, 128]
        channels = [max(4, round(c * width / 4) * 4) for c in base]
        strides = strides or [2, 1, 2, 1, 2, 1]
        if len(channels) != len(strides) + 1 or any(s not in (1, 2) for s in strides):
            raise ValueError("Provide one stride (1 or 2) for each block channel")
        self.model_config = dict(
            num_classes=num_classes,
            dropout=dropout,
            width=width,
            depth=depth,
            channels=base,
            strides=strides,
            version=version,
            expansion=expansion,
            residual=residual,
        )
        repeats = stage_depths if stage_depths is not None else [depth] * len(strides)
        if len(repeats) != len(strides) or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in repeats
        ):
            raise ValueError("stage_depths must contain one positive integer per stage")
        if residual_stages is not None and (
            len(set(residual_stages)) != len(residual_stages)
            or any(not isinstance(i, int) or i < 0 or i >= len(strides) for i in residual_stages)
        ):
            raise ValueError("residual_stages must contain unique existing stage indices (zero-based)")
        # Omit default options: legacy model_config dictionaries remain exactly unchanged.
        optional = dict(
            kernel_size=kernel_size,
            activation=activation,
            stage_depths=stage_depths,
            residual_stages=residual_stages,
            projection_activation=projection_activation,
        )
        defaults = dict(
            kernel_size=3,
            activation="relu",
            stage_depths=None,
            residual_stages=None,
            projection_activation="post",
        )
        self.model_config.update({k: v for k, v in optional.items() if v != defaults[k]})
        layers = [conv_bn_relu(3, channels[0], stride=2, activation=activation)]
        cin = channels[0]
        for stage, (cout, stride, n) in enumerate(zip(channels[1:], strides, repeats)):
            for repeat in range(n):
                s = stride if repeat == 0 else 1
                block = (
                    GumiBlockV1(cin, cout, s, kernel_size, activation)
                    if version == "v1"
                    else GumiBlockV2(
                        cin,
                        cout,
                        s,
                        expansion,
                        residual and (residual_stages is None or stage in residual_stages),
                        kernel_size,
                        activation,
                        projection_activation,
                    )
                )
                block.stage_index, block.repeat_index = stage, repeat
                layers.append(block)
                cin = cout
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(cin, num_classes)
        self.default_input_size = 160

    def forward(self, x):
        return self.classifier(self.dropout(torch.flatten(self.pool(self.features(x)), 1)))


class GumiNetV2(GumiNetV1):
    def __init__(self, num_classes=1000, expansion=1.0, residual=False, **kwargs):
        kwargs.pop("version", None)
        super().__init__(num_classes, version="v2", expansion=expansion, residual=residual, **kwargs)


def guminet_nano(**kwargs):
    return GumiNetV1(**(dict(width=0.5, depth=1) | kwargs))


def guminet_tiny(**kwargs):
    return GumiNetV1(**(dict(width=1.0, depth=1) | kwargs))


def guminet_small(**kwargs):
    return GumiNetV1(**(dict(width=1.25, depth=2) | kwargs))


def _v2(default_expansion, default_residual, **kwargs):
    # Repeated stages allow identity residuals without projected shortcut branches.
    return GumiNetV2(**(dict(depth=2, expansion=default_expansion, residual=default_residual) | kwargs))


MODEL_REGISTRY = {
    "v2_refined_d1_w075": lambda **kw: _v2(1.0, False, **(dict(depth=1, width=0.75) | kw)),
    "v2_candidate_stage3": lambda **kw: _v2(
        1.0, True, **(dict(depth=1, width=0.75, stage_depths=[1, 1, 1, 2, 1, 1], residual_stages=[3]) | kw)
    ),
    "guminet_v1": GumiNetV1,
    "guminet_nano": guminet_nano,
    "guminet_tiny": guminet_tiny,
    "guminet_small": guminet_small,
    "guminet_v2_plain": lambda **kw: _v2(1.0, False, **kw),
    "guminet_v2_residual": lambda **kw: _v2(1.0, True, **kw),
    "guminet_v2_low_expansion": lambda **kw: _v2(1.5, False, **kw),
    "guminet_v2_low_expansion_residual": lambda **kw: _v2(1.5, True, **kw),
}


def create_model(name="guminet_tiny", num_classes=1000, pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("Pretrained GumiNet weights are not available. Supply a trained checkpoint.")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
    model.architecture = "guminet"  # Exact structural kwargs, rather than alias, are saved.
    return model


def count_parameters(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if not trainable or p.requires_grad)


def model_architecture_summary(model, **kwargs):
    from .analysis import model_architecture_summary as summarize

    return summarize(model, **kwargs)

"""Non-mutating structural and gradient diagnostics, never classification evidence."""

import copy

import torch
from torch import nn

from . import GumiBlockV1, GumiBlockV2, count_parameters


def model_architecture_summary(model, input_size=128, batch_size=1, include_gradients=False):
    from guminet.benchmark import estimate_macs

    # Copy preserves weights, buffers, heterogeneous training modes, device and hooks.
    probe = copy.deepcopy(model).cpu().eval()
    blocks, handles, inputs = [], [], {}
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(0)
        example = torch.randn(
            batch_size, 3, input_size, input_size, requires_grad=include_gradients, generator=generator
        )
        costs = estimate_macs(probe, example.detach())

        def hook(name):
            def observe(module, args, output):
                x = args[0]
                if include_gradients:
                    x.retain_grad()
                    inputs[name] = x
                blocks.append(
                    dict(
                        name=name,
                        stage=getattr(module, "stage_index", None),
                        repeat=getattr(module, "repeat_index", None),
                        type=type(module).__name__,
                        active_residual=bool(getattr(module, "use_residual", False)),
                        input_shape=list(x.shape),
                        output_shape=list(output.shape),
                        output_bytes=output.numel() * output.element_size(),
                        feature_mean=float(output.detach().mean()),
                        feature_variance=float(output.detach().var(unbiased=False)),
                        zero_fraction=float((output.detach() == 0).float().mean()),
                    )
                )

            return observe

        try:
            for name, module in probe.named_modules():
                if isinstance(module, (GumiBlockV1, GumiBlockV2)) or name == "features.0":
                    handles.append(module.register_forward_hook(hook(name)))
            with torch.set_grad_enabled(include_gradients):
                logits = probe(example)
                if include_gradients:
                    logits.square().mean().backward()
                    for block in blocks:
                        grad = inputs[block["name"]].grad
                        block["input_gradient_norm"] = float(grad.norm()) if grad is not None else None
                        block["finite_gradient"] = grad is not None and bool(torch.isfinite(grad).all())
        finally:
            for handle in handles:
                handle.remove()
    residuals = [b["name"] for b in blocks if b["active_residual"]]
    return dict(
        blocks=blocks,
        active_residual_blocks=len(residuals),
        active_residual_names=residuals,
        total_parameters=count_parameters(model),
        trainable_parameters=count_parameters(model, True),
        input_shape=list(example.shape),
        output_shape=list(logits.shape),
        feature_map_sizes=[b["output_shape"] for b in blocks],
        estimated_activation_bytes=sum(b["output_bytes"] for b in blocks),
        max_block_output_bytes=max((b["output_bytes"] for b in blocks), default=0),
        activation_estimate_definition="Sum of stem and block output tensors; not live peak, saved backward tensors, or RSS",
        **costs,
    )


def residual_path_probe(block, channels=24, size=8):
    """Zero the learned branch on a copy to isolate the identity's input derivative."""
    probe = copy.deepcopy(block).cpu().eval()
    for p in probe.parameters():
        nn.init.zeros_(p)
    x = torch.ones(2, channels, size, size, requires_grad=True)
    y = probe(x)
    grad = torch.autograd.grad(y.sum(), x)[0]
    return dict(
        active_residual=probe.use_residual,
        identity_output=bool(torch.equal(y, x)),
        gradient_min=float(grad.min()),
        gradient_max=float(grad.max()),
        finite=bool(torch.isfinite(grad).all()),
        diagnostic="Zero-branch identity derivative; not accuracy evidence",
    )

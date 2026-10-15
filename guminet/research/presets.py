"""One-factor controls; moving a repeat block is one residual-location design factor."""

import copy


def experiment_presets(group):
    if group == "v1":
        base = dict(model="guminet_tiny", model_kwargs={}, input_size=128)
        changes = {
            "kernel5": dict(kernel_size=5),
            "late_downsampling": dict(strides=[1, 1, 2, 2, 2, 1]),
            "balanced_channels": dict(channels=[16, 24, 40, 56, 72, 88, 112]),
            "late_channels": dict(channels=[16, 20, 28, 40, 56, 96, 144]),
            "leaky_relu": dict(activation="leaky_relu"),
        }
    elif group == "v2":
        base = dict(
            model="v2_refined_d1_w075",
            model_kwargs=dict(
                depth=1, width=0.75, expansion=1.0, residual=False, stage_depths=[1, 1, 1, 1, 1, 1]
            ),
            input_size=128,
        )
        changes = {
            "single_repeat_control": dict(stage_depths=[1, 1, 1, 2, 1, 1]),
            "candidate_residual": dict(stage_depths=[1, 1, 1, 2, 1, 1], residual=True),
            "alternate_residual_location": dict(stage_depths=[1, 2, 1, 1, 1, 1], residual=True),
            "late_downsampling": dict(strides=[1, 1, 2, 2, 2, 1]),
            "balanced_channels": dict(channels=[16, 24, 40, 56, 72, 88, 112]),
            "projection_linear": dict(projection_activation="linear"),
            "projection_pre": dict(projection_activation="pre"),
        }
    elif group == "training":
        base = dict(model="guminet_tiny", model_kwargs={}, input_size=128)
        return {
            "baseline": base,
            "lr": dict(copy.deepcopy(base), training_overrides=dict(lr=0.0005)),
            "scheduler": dict(copy.deepcopy(base), training_overrides=dict(scheduler="step")),
            "augmentation": dict(copy.deepcopy(base), training_overrides=dict(augmentation=False)),
            "batch_size": dict(copy.deepcopy(base), training_overrides=dict(batch_size=8)),
        }
    else:
        raise ValueError("group must be legacy, v1, v2 or training")
    result = {"baseline": base}
    for name, kw in changes.items():
        entry = copy.deepcopy(base)
        entry["model_kwargs"].update(kw)
        entry["control"] = "baseline"
        result[name] = entry
    if group == "v2":
        # These pairs isolate skip addition and stage location; do not compare everything as a single factor.
        result["candidate_residual"]["control"] = "single_repeat_control"
        result["alternate_residual_location"]["control"] = "candidate_residual"
        for name in ("projection_linear", "projection_pre"):
            result[name]["model_kwargs"].update(stage_depths=[1, 1, 1, 2, 1, 1], residual=True)
            result[name]["control"] = "candidate_residual"
    return result

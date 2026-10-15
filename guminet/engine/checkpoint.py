"""Versioned, weights-only loading; no unrestricted pickle fallback."""

import os
from pathlib import Path

import torch

from guminet.models import GumiNetV1


def load_checkpoint(path, device="cpu"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}; no pretrained weights are bundled")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"format_version", "model_config", "model_state", "class_to_idx", "input_size", "mean", "std"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("Invalid or incomplete GumiNet checkpoint metadata")
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    mapping = checkpoint["class_to_idx"]
    if not isinstance(mapping, dict) or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Invalid class mapping")
    if len(mapping) != checkpoint["model_config"]["num_classes"]:
        raise ValueError("Classifier size does not match class mapping")
    if checkpoint["input_size"] < 16 or len(checkpoint["mean"]) != 3 or len(checkpoint["std"]) != 3:
        raise ValueError("Invalid preprocessing metadata")
    if any(s <= 0 for s in checkpoint["std"]):
        raise ValueError("Invalid normalization standard deviations")
    return checkpoint


def model_from_checkpoint(path, device="cpu"):
    checkpoint = load_checkpoint(path, device)
    architecture = checkpoint.get("architecture", "guminet")
    if architecture == "guminet":
        model = GumiNetV1(**checkpoint["model_config"])
    elif architecture == "comparison":
        from guminet.benchmark import create_comparison_model

        model = create_comparison_model(**checkpoint["model_config"], pretrained=False)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device), checkpoint


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

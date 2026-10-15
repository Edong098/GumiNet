"""Generate tiny learnable RGB classes; functional checks only, never accuracy evidence."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate(root, samples=8, classes=3, size=48, seed=42):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    rng = np.random.default_rng(seed)
    for split in ("train", "val", "test"):
        for label in range(classes):
            path = root / split / f"class_{label}"
            path.mkdir(parents=True, exist_ok=True)
            for i in range(samples):
                pixels = rng.integers(0, 35, size=(size, size, 3), dtype=np.uint8)
                pixels[:, :, label % 3] += 180
                Image.fromarray(pixels).save(path / f"{i}.png")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    generate(args.output, args.samples)

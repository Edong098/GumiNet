import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def dataset_root(tmp_path):
    rng = np.random.default_rng(8)
    root = tmp_path / "data"
    for split in ("train", "val", "test"):
        for label in ("a", "b"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(3):
                pixels = rng.integers(0, 30, (32, 32, 3), dtype=np.uint8)
                pixels[:, :, int(label == "b")] += 180
                Image.fromarray(pixels).save(folder / f"{i}.png")
    return root

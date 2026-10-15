import shutil

import pytest
import torch
from PIL import Image

from guminet.data import FolderDataset, ImageTransform, check_leakage, make_loaders, stratified_split


def test_lazy_data(dataset_root):
    loaders = make_loaders(dataset_root, input_size=32, batch_size=2)
    assert loaders["train"].dataset.summary() == {"a": 3, "b": 3}
    x, y = next(iter(loaders["val"]))
    assert x.shape == (2, 3, 32, 32)
    assert y.tolist() == [0, 0]
    val = loaders["val"].dataset
    assert torch.equal(val[0][0], val[0][0])
    assert ImageTransform(32)(Image.new("L", (20, 20))).shape == (3, 32, 32)


def test_corrupt_and_mapping(dataset_root):
    (dataset_root / "train/a/corrupt.png").write_bytes(b"not an image")
    ds = FolderDataset(dataset_root / "train")
    assert len(ds.corrupted) == 1 and len(ds) == 6
    with pytest.raises(ValueError, match="Inconsistent"):
        FolderDataset(dataset_root / "train", class_to_idx={"a": 0})


def test_leakage(dataset_root):
    shutil.copy(dataset_root / "train/a/0.png", dataset_root / "val/a/copy.png")
    with pytest.raises(ValueError, match="leakage"):
        check_leakage({split: FolderDataset(dataset_root / split) for split in ("train", "val")})


def test_split(dataset_root, tmp_path):
    destination = stratified_split(dataset_root / "train", tmp_path / "split", ratios=(0.34, 0.33, 0.33))
    loaders = make_loaders(destination, input_size=32)
    assert sum(len(loader.dataset) for loader in loaders.values()) == 6
    with pytest.raises(FileExistsError):
        stratified_split(dataset_root / "train", destination)


def test_tuning_loader_ignores_test(dataset_root):
    # Test split may not even be usable; tuning must not inspect it.
    shutil.rmtree(dataset_root / "test/b")
    loaders = make_loaders(dataset_root, input_size=32, include_test=False)
    assert set(loaders) == {"train", "val"}

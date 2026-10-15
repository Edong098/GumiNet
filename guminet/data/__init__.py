"""Lazy folder datasets with shared labels and deterministic evaluation preprocessing."""

import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageTransform:
    def __init__(self, size=160, train=False, mean=MEAN, std=STD):
        if size < 16 or len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError("Image size >= 16 and three positive standard deviations required")
        self.size, self.train, self.mean, self.std = size, train, tuple(mean), tuple(std)

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        arr = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]


class FolderDataset(Dataset):
    def __init__(self, root, transform=None, class_to_idx=None, verify=True):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = (
            dict(class_to_idx) if class_to_idx is not None else dict(zip(classes, range(len(classes))))
        )
        if not self.class_to_idx or set(classes) != set(self.class_to_idx):
            raise ValueError(f"Inconsistent class folders at {root}: {classes}")
        if sorted(self.class_to_idx.values()) != list(range(len(self.class_to_idx))):
            raise ValueError("Class indices must be contiguous starting at zero")
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.transform = transform or ImageTransform()
        self.samples, self.corrupted = [], []
        for label in self.classes:
            paths = sorted(p for p in (self.root / label).rglob("*") if p.suffix.lower() in EXTENSIONS)
            valid_count = 0
            for path in paths:
                if verify:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                    except (OSError, ValueError, Image.DecompressionBombError):
                        self.corrupted.append(str(path))
                        continue
                self.samples.append((path, self.class_to_idx[label]))
                valid_count += 1
            if not valid_count:
                raise ValueError(f"No valid images in class {label} at {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image), label

    def summary(self):
        return {
            name: sum(label == idx for _, label in self.samples) for name, idx in self.class_to_idx.items()
        }


def check_leakage(datasets):
    """Streaming SHA256 checks exact file duplicates across splits; no decoded-image cache."""
    seen = {}
    for split, dataset in datasets.items():
        for path, _ in dataset.samples:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            key = digest.digest()
            if key in seen and seen[key][0] != split:
                raise ValueError(f"Data leakage between {seen[key]} and {(split, str(path))}")
            seen[key] = (split, str(path))


def make_loaders(
    root,
    input_size=160,
    batch_size=4,
    num_workers=0,
    seed=42,
    mean=MEAN,
    std=STD,
    leakage_check=True,
    include_test=True,
):
    root = Path(root)
    datasets = {}
    for split in ("train", "val", "test") if include_test else ("train", "val"):
        if not (root / split).exists() and split == "test":
            continue
        datasets[split] = FolderDataset(
            root / split,
            ImageTransform(input_size, split == "train", mean, std),
            datasets["train"].class_to_idx if datasets else None,
        )
    if leakage_check:
        check_leakage(datasets)
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            generator=generator if split == "train" else None,
            worker_init_fn=seed_worker,
        )
        for split, ds in datasets.items()
    }


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def stratified_split(source, destination, ratios=(0.7, 0.15, 0.15), seed=42):
    """Copy files into a NEW destination; retain source and require each class in each split."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("Three positive ratios summing to one required")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must not overlap")
    if destination.exists():
        raise FileExistsError(destination)
    dataset = FolderDataset(source)
    rng = random.Random(seed)
    plan = []
    for name, idx in dataset.class_to_idx.items():
        paths = [p for p, label in dataset.samples if label == idx]
        if len(paths) < 3:
            raise ValueError(f"Class {name} needs at least three images")
        rng.shuffle(paths)
        ntrain = max(1, min(len(paths) - 2, round(len(paths) * ratios[0])))
        nval = max(1, min(len(paths) - ntrain - 1, round(len(paths) * ratios[1])))
        for split, selected in zip(
            ("train", "val", "test"), (paths[:ntrain], paths[ntrain : ntrain + nval], paths[ntrain + nval :])
        ):
            for i, path in enumerate(selected):
                plan.append((path, destination / split / name / f"{i:08d}{path.suffix.lower()}"))
    for path, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination

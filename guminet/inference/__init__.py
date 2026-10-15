import time

import torch
from PIL import Image

from guminet.data import ImageTransform
from guminet.engine.checkpoint import model_from_checkpoint
from guminet.utils import resolve_device


class GumiPredictor:
    def __init__(self, model, metadata, device="auto"):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.metadata = metadata
        self.classes = sorted(metadata["class_to_idx"], key=metadata["class_to_idx"].get)
        self.transform = ImageTransform(metadata["input_size"], False, metadata["mean"], metadata["std"])

    @classmethod
    def from_checkpoint(cls, path, device="auto"):
        model, metadata = model_from_checkpoint(path)
        return cls(model, metadata, device)

    def predict(self, image):
        return self.predict_batch([image])[0]

    @torch.inference_mode()
    def predict_batch(self, images, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results = []
        for offset in range(0, len(images), batch_size):
            tensors = []
            for image in images[offset : offset + batch_size]:
                if isinstance(image, Image.Image):
                    tensors.append(self.transform(image))
                else:
                    with Image.open(image) as opened:
                        tensors.append(self.transform(opened))
            x = torch.stack(tensors).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            probabilities = self.model(x).softmax(1).cpu().tolist()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start
            for p in probabilities:
                idx = max(range(len(p)), key=p.__getitem__)
                results.append(
                    dict(
                        class_name=self.classes[idx],
                        class_index=idx,
                        confidence=p[idx],
                        probabilities=dict(zip(self.classes, p)),
                        inference_ms_per_image=elapsed * 1000 / len(tensors),
                    )
                )
        return results

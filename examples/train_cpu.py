"""Run from checkout after: python scripts/make_synthetic.py dataset."""

import torch

from guminet import GumiTrainer, create_model
from guminet.data import make_loaders
from guminet.utils import seed_everything

torch.set_num_threads(2)
seed_everything(42)  # Set BEFORE constructing the model.
loaders = make_loaders("dataset", input_size=128, batch_size=4)
model = create_model("guminet_nano", num_classes=len(loaders["train"].dataset.classes))
trainer = GumiTrainer(
    model, {"epochs": 2, "output_dir": "runs/example"}, loaders["train"].dataset.class_to_idx
)
trainer.fit(loaders["train"], loaders["val"])

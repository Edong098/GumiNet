"""GumiNet: experimental lightweight convolutional neural networks."""

__version__ = "0.1.0"

from .engine import GumiTrainer
from .inference import GumiPredictor
from .models import GumiNetV1, GumiNetV2, create_model, guminet_nano, guminet_small, guminet_tiny

__all__ = [
    "GumiNetV1",
    "GumiNetV2",
    "create_model",
    "guminet_nano",
    "guminet_tiny",
    "model_architecture_summary",
    "guminet_small",
    "GumiTrainer",
    "GumiPredictor",
]

from .models import model_architecture_summary

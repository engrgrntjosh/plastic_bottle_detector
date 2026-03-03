"""Plastic Bottle Detector package.

A production-ready YOLOv8-based detector for plastic bottles.

Public API:
    - prepare_data(): Download dataset and generate data.yaml.
    - train_model(): Train YOLOv8 with optional W&B tracking.
    - evaluate_model(): Evaluate trained model on validation set.
    - load_and_infer(): Load model and run inference.
"""

__version__ = "0.1.0"

__all__ = [
    "prepare_data",
    "train_model",
    "evaluate_model",
    "load_and_infer",
    "__version__",
]


def __getattr__(name: str):
    """Lazy-import public API to avoid loading heavy dependencies at import time."""
    if name == "prepare_data":
        from .data import prepare_data
        return prepare_data
    if name == "train_model":
        from .train import train_model
        return train_model
    if name == "evaluate_model":
        from .evaluate import evaluate_model
        return evaluate_model
    if name == "load_and_infer":
        from .infer import load_and_infer
        return load_and_infer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Plastic Bottle Detector package.

This package provides a production-ready YOLOv8-based detector for plastic bottles.

Public API:

    - prepare_data()

    - train_model()

    - evaluate_model()

    - load_and_infer()

"""

__version__ = "0.1.0"  


from .data import prepare_data

from .train import train_model

from .evaluate import evaluate_model

from .infer import load_and_infer

__all__ = [

    "prepare_data",

    "train_model",

    "evaluate_model",

    "load_and_infer",

    "__version__",

]

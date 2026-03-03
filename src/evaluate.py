"""Model evaluation on validation/test sets."""

import structlog
from ultralytics import YOLO

logger = structlog.get_logger(__name__)


def evaluate_model(weights: str, data_yaml: str, imgsz: int = 640) -> dict:
    """Evaluate a trained YOLO model.

    Args:
        weights: Path to model weights (.pt file).
        data_yaml: Path to data.yaml config.
        imgsz: Image size for evaluation.

    Returns:
        Dictionary of evaluation metrics.
    """
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz)
    logger.info("evaluation_completed", metrics=metrics.results_dict)
    return metrics.results_dict

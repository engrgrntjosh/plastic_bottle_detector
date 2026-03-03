"""Inference utilities for running detection on images/video."""

from typing import Any, List

import structlog
import torch
from ultralytics import YOLO

from src.config import InferConfig

logger = structlog.get_logger(__name__)


def run_inference(model: YOLO, source: str, conf: float) -> List[Any]:
    """Run detection inference on the given source.

    Args:
        model: Loaded YOLO model.
        source: Path to image, directory, or video.
        conf: Confidence threshold for detections.

    Returns:
        List of numpy arrays containing detection boxes, one per frame.
    """
    try:
        results = model(source, conf=conf, stream=True)
        detections = [r.boxes.data.cpu().numpy() for r in results]
        logger.info("inference_completed", num_frames=len(detections))
        return detections
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_and_infer(config: InferConfig) -> List[Any]:
    """Load a model from config and run inference.

    Args:
        config: Inference configuration with weights, source, and confidence.

    Returns:
        List of detection arrays.
    """
    model = YOLO(config.weights)
    return run_inference(model, config.source, config.conf)

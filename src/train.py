"""YOLOv8 model training with optional W&B logging."""

import structlog
import torch
from ultralytics import YOLO

from src.config import AppConfig
from src.utils import get_device

logger = structlog.get_logger(__name__)


def _init_wandb(config: AppConfig) -> bool:
    """Attempt to initialize Weights & Biases. Returns True on success."""
    try:
        import wandb

        wandb.init(
            project=config.train.project,
            name=config.train.name,
            config=config.model_dump(),
        )
        return True
    except Exception as e:
        logger.warning("wandb_init_skipped", reason=str(e))
        return False


def _finish_wandb(success: bool) -> None:
    """Finalize the W&B run."""
    try:
        import wandb

        wandb.finish(exit_code=0 if success else 1)
    except Exception:
        pass


def train_model(config: AppConfig) -> dict:
    """Train a YOLOv8 model.

    Supports multi-GPU via device config and optional W&B experiment tracking.

    Args:
        config: Full application configuration.

    Returns:
        Dictionary of training results and metrics.

    Raises:
        Exception: If training fails for any reason.
    """
    wandb_active = _init_wandb(config)
    device = get_device(config)
    success = False

    try:
        model = YOLO(config.train.model)
        results = model.train(
            data=config.data.yaml_path,
            epochs=config.train.epochs,
            batch=config.train.batch,
            imgsz=config.train.imgsz,
            device=device,
            amp=True,
            project=config.train.project,
            name=config.train.name,
            exist_ok=True,
        )
        success = True
        logger.info("training_completed", results=results.results_dict)
        return results.results_dict
    except Exception:
        logger.exception("training_failed")
        raise
    finally:
        if wandb_active:
            _finish_wandb(success)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

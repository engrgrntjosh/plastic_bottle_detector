"""Pipeline entrypoint — download data, train, and evaluate."""

import hydra
import structlog
from omegaconf import DictConfig

from src.config import AppConfig
from src.data import prepare_data
from src.evaluate import evaluate_model
from src.train import train_model
from src.utils import setup_logging

logger = structlog.get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the full training pipeline."""
    config = AppConfig(**cfg)
    setup_logging(config.log_level)

    logger.info("pipeline_started")
    prepare_data(config.data)
    train_model(config)
    evaluate_model(config.infer.weights, config.data.yaml_path, config.train.imgsz)
    logger.info("pipeline_completed")


if __name__ == "__main__":
    main()

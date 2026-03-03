"""Logging and device utilities."""

import logging

import structlog
import torch

from src.config import AppConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(level=level.upper(), force=True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    structlog.get_logger().info("logging_configured", level=level)


def get_device(config: AppConfig) -> str:
    """Resolve the device string for YOLO training.

    Args:
        config: Application configuration.

    Returns:
        Device string: 'cpu', 'cuda', or comma-separated GPU IDs like '0,1'.
    """
    if config.train.device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return ",".join(map(str, config.train.device))

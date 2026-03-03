"""Dataset download and preparation utilities."""

import os
import subprocess

import structlog
import yaml

from src.config import DataConfig

logger = structlog.get_logger(__name__)


def download_dataset(config: DataConfig) -> None:
    """Download the dataset from Kaggle and extract it.

    Requires a valid Kaggle API token at ~/.kaggle/kaggle.json.

    Args:
        config: Data configuration with dataset slug and target directory.

    Raises:
        RuntimeError: If the Kaggle CLI download fails.
    """
    if os.path.exists(config.data_dir) and os.listdir(config.data_dir):
        logger.info("dataset_directory_exists", path=config.data_dir)
        return

    os.makedirs(config.data_dir, exist_ok=True)

    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", config.kaggle_dataset,
            "-p", config.data_dir,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        dataset_name = config.kaggle_dataset.split("/")[-1]
        zip_path = os.path.join(config.data_dir, f"{dataset_name}.zip")
        subprocess.run(["unzip", "-q", zip_path, "-d", config.data_dir], check=True)
        os.remove(zip_path)

        logger.info("dataset_downloaded", path=config.data_dir)
    except subprocess.CalledProcessError as e:
        logger.error(
            "dataset_download_failed",
            error=str(e),
            stdout=e.stdout.decode(),
            stderr=e.stderr.decode(),
        )
        raise RuntimeError(
            "Dataset download failed. Ensure Kaggle API credentials are configured."
        ) from e


def create_data_yaml(config: DataConfig) -> None:
    """Generate the YOLO-format data.yaml file.

    Assumes dataset layout:
        data/Plastic Bottle Image Dataset/{train,valid,test}/{images,labels}

    Args:
        config: Data configuration with paths.
    """
    data_dict = {
        "path": os.path.abspath(config.data_dir),
        "train": "Plastic Bottle Image Dataset/train/images",
        "val": "Plastic Bottle Image Dataset/valid/images",
        "test": "Plastic Bottle Image Dataset/test/images",
        "nc": 1,
        "names": ["bottle"],
    }

    os.makedirs(os.path.dirname(config.yaml_path) or ".", exist_ok=True)
    with open(config.yaml_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    logger.info("data_yaml_created", path=config.yaml_path)


def prepare_data(config: DataConfig) -> None:
    """Download dataset and generate data.yaml."""
    download_dataset(config)
    create_data_yaml(config)

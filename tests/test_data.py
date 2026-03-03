import os

import pytest
import yaml

from src.config import DataConfig
from src.data import create_data_yaml, download_dataset


@pytest.fixture()
def temp_dir(tmp_path):
    yield str(tmp_path)


def test_download_dataset_fails_without_credentials(temp_dir):
    config = DataConfig(data_dir=temp_dir)
    with pytest.raises(RuntimeError):
        download_dataset(config)


def test_create_data_yaml(temp_dir):
    yaml_path = os.path.join(temp_dir, "test.yaml")
    config = DataConfig(data_dir=temp_dir, yaml_path=yaml_path)
    create_data_yaml(config)
    assert os.path.exists(yaml_path)


def test_create_data_yaml_contents(temp_dir):
    yaml_path = os.path.join(temp_dir, "test.yaml")
    config = DataConfig(data_dir=temp_dir, yaml_path=yaml_path)
    create_data_yaml(config)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    assert data["nc"] == 1
    assert data["names"] == ["bottle"]
    assert "train" in data
    assert "val" in data

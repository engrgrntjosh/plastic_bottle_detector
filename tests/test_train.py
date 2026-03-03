import os

import pytest

from src.config import AppConfig, DataConfig, InferConfig, TrainConfig
from src.train import train_model


@pytest.mark.integration
def test_train_model(tmp_path):
    config = AppConfig(
        data=DataConfig(yaml_path="tests/dummy_data.yaml"),
        train=TrainConfig(epochs=1, batch=2, imgsz=320, project=str(tmp_path)),
        infer=InferConfig(),
    )
    train_model(config)
    weights_path = os.path.join(
        config.train.project, config.train.name, "weights", "last.pt"
    )
    assert os.path.exists(weights_path)

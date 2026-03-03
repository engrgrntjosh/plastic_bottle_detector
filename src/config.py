"""Pydantic configuration models for the detection pipeline."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for dataset handling."""

    kaggle_dataset: str = Field(
        default="siddharthkumarsah/plastic-bottles-image-dataset",
        description="Kaggle dataset slug",
    )
    data_dir: str = Field(default="./data", description="Path to store dataset")
    yaml_path: str = Field(default="./data/data.yaml", description="Path to YOLO data config")


class TrainConfig(BaseModel):
    """Configuration for training."""

    model: str = Field(default="yolov8n.pt", description="Pretrained YOLOv8 model")
    epochs: int = Field(default=50, description="Number of training epochs")
    batch: int = Field(default=16, description="Batch size")
    imgsz: int = Field(default=640, description="Image size")
    device: Optional[List[int]] = Field(default=None, description="GPU devices, e.g., [0,1]")
    project: str = Field(default="runs/train", description="Project directory")
    name: str = Field(default="plastic_bottle_detector", description="Experiment name")


class InferConfig(BaseModel):
    """Configuration for inference."""

    weights: str = Field(
        default="runs/train/plastic_bottle_detector/weights/best.pt",
        description="Path to trained weights",
    )
    conf: float = Field(default=0.5, description="Confidence threshold")
    source: str = Field(default="test_images", description="Inference source (dir, file, URL)")


class AppConfig(BaseModel):
    """Root configuration."""

    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    infer: InferConfig = InferConfig()
    log_level: str = Field(default="INFO", description="Logging level")

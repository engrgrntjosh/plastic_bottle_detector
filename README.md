# Plastic Bottle Detector

A YOLOv8-based object detection pipeline for detecting plastic bottles in images and video. Built with Ultralytics, Hydra for configuration, and Weights & Biases for experiment tracking.

## Project Structure

```
plastic_bottle_detector/
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── main.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_train.py
├── .gitignore
├── setup.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- [Kaggle API credentials](https://www.kaggle.com/docs/api) in `~/.kaggle/kaggle.json`
- (Optional) CUDA-compatible GPU for training
- (Optional) [Weights & Biases](https://wandb.ai/) account for experiment tracking

## Installation

```bash
git clone https://github.com/<your-username>/plastic_bottle_detector.git
cd plastic_bottle_detector
pip install -e ".[dev]"
```

Or using requirements.txt directly:

```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline

```bash
python -m src.main
```

### Override Config via CLI (Hydra)

```bash
python -m src.main train.epochs=100 train.batch=32 train.device=[0,1]
```

### Inference Only

```bash
python -m src.infer
```

## Configuration

All settings live in `configs/default.yaml` and can be overridden via CLI. See `src/config.py` for schema details.

## Testing

```bash
pytest tests/ -v
pytest tests/ -v -m integration  # requires dataset and GPU
```

## License

MIT

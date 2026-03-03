from setuptools import setup, find_packages

setup(
    name="plastic-bottle-detector",
    version="0.1.0",
    description="YOLOv8-based plastic bottle detection pipeline",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "pydantic>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "structlog>=23.0.0",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "hypothesis>=6.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
)

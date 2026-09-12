"""Dataset Download and Preparation Script for RT-DETR PPE Detection.

Prepares the dataset structure for training and evaluation.
Supports downloading directly from Roboflow Universe or extracting a local archive.
"""

import os
import sys
import yaml
from pathlib import Path
from loguru import logger

DATA_DIR = Path("data")
YAML_PATH = DATA_DIR / "ppe_data.yaml"

TARGET_CLASSES = [
    "person",
    "hard-hat",
    "no-helmet",
    "safety-vest",
    "no-vest"
]


def create_dataset_directories():
    """Create standard YOLO / RT-DETR directory hierarchy."""
    for split in ["train", "val", "test"]:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    logger.info("Created data directory hierarchy in data/images/ and data/labels/")


def generate_dataset_yaml(base_dir: str = "data"):
    """Generate data.yaml file required by Ultralytics RT-DETR."""
    data_cfg = {
        "path": os.path.abspath(base_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False)

    logger.info(f"Generated dataset configuration file at: {YAML_PATH}")
    logger.info(f"Target classes ({len(TARGET_CLASSES)}): {TARGET_CLASSES}")


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int):
    """Download directly using Roboflow Python SDK if an API key is supplied."""
    try:
        from roboflow import Roboflow
        logger.info(f"Connecting to Roboflow workspace '{workspace}', project '{project}'...")
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8", location=str(DATA_DIR))
        logger.info(f"Roboflow dataset downloaded successfully to {DATA_DIR}")
        return True
    except Exception as e:
        logger.warning(f"Roboflow download failed: {e}. You can download the zip manually or use sample verification set.")
        return False


def main():
    logger.info("Initializing PPE Dataset Preparation Pipeline...")
    create_dataset_directories()
    generate_dataset_yaml()

    roboflow_key = os.getenv("ROBOFLOW_API_KEY")
    if roboflow_key:
        download_roboflow_dataset(
            api_key=roboflow_key,
            workspace="safety-first",
            project="hard-hat-worker-safety-equipments",
            version=1
        )
    else:
        logger.info(
            "ROBOFLOW_API_KEY environment variable not set. "
            "Created directory structure and 'data/ppe_data.yaml'. "
            "To train on Google Colab or locally, place your images and label files in 'data/images/' and 'data/labels/'."
        )


if __name__ == "__main__":
    main()

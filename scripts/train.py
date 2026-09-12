"""Fine-tune RT-DETR model for PPE safety detection."""

import os
import sys
import time
import json
import random
import platform
import shutil
from pathlib import Path
import numpy as np
import torch
from loguru import logger

from src.config import settings


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seed set to {seed}")


def gather_system_environment() -> dict:
    env_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    return env_info


def train_rtdetr(
    data_yaml: str = "data/ppe_data.yaml",
    base_model: str = "rtdetr-l.pt",
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    learning_rate: float = 0.0001,
    seed: int = 42,
    project_name: str = "runs/train",
    run_name: str = "rtdetr_ppe_run",
):
    """Execute RT-DETR fine-tuning with full telemetry logging."""
    from ultralytics import RTDETR

    set_seed(seed)
    env_info = gather_system_environment()
    logger.info(f"System Environment: {env_info}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Dataset configuration '{data_yaml}' not found. Run scripts/download_dataset.py first."
        )

    # Auto-adjust batch size if on CPU or low VRAM
    if not torch.cuda.is_available():
        logger.warning("CUDA not detected. Lowering batch size to 4 for CPU execution.")
        batch_size = 4

    hyperparameters = {
        "base_model": base_model,
        "dataset_yaml": data_yaml,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate_initial": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "seed": seed,
        "target_classes": settings.TARGET_CLASSES,
        "unified_confidence_threshold": settings.CONFIDENCE_THRESHOLD,
    }

    # Save hyperparameters before training starts
    os.makedirs(project_name, exist_ok=True)
    hyperparams_file = os.path.join(project_name, "training_hyperparameters.json")
    with open(hyperparams_file, "w") as f:
        json.dump({"environment": env_info, "hyperparameters": hyperparameters}, f, indent=2)
    logger.info(f"Saved run hyperparameters and hardware audit to: {hyperparams_file}")

    # Initialize RT-DETR
    logger.info(f"Initializing RT-DETR checkpoint: {base_model}")
    model = RTDETR(base_model)

    start_time = time.time()

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        lr0=learning_rate,
        optimizer="AdamW",
        weight_decay=0.0001,
        seed=seed,
        device=0 if torch.cuda.is_available() else "cpu",
        project=project_name,
        name=run_name,
        save=True,
        plots=True,
        verbose=True,
    )

    elapsed_time_sec = time.time() - start_time
    hours, rem = divmod(elapsed_time_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s ({elapsed_time_sec:.1f}s total)"
    logger.info(f"Training completed successfully in {duration_str}")

    # Copy best weights to standard weights location
    best_weights_src = Path(project_name) / run_name / "weights" / "best.pt"
    dest_weights = Path(settings.MODEL_PATH)
    dest_weights.parent.mkdir(parents=True, exist_ok=True)

    if best_weights_src.exists():
        shutil.copy(best_weights_src, dest_weights)
        logger.info(f"Fine-tuned weights successfully deployed to: {dest_weights}")
    else:
        logger.warning(f"Could not locate best.pt at {best_weights_src}")

    # Record final training summary
    summary = {
        "environment": env_info,
        "hyperparameters": hyperparameters,
        "training_duration": duration_str,
        "training_seconds": elapsed_time_sec,
        "final_weights_path": str(dest_weights),
    }
    with open(os.path.join(project_name, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Reproducibility artifacts and training summary exported successfully.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproducible RT-DETR Training for PPE Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (e.g. 16 on GPU, 4 on CPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data", type=str, default="data/ppe_data.yaml", help="Path to data YAML")

    args = parser.parse_args()

    train_rtdetr(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        learning_rate=args.lr,
        seed=args.seed,
    )

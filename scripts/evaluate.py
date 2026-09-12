"""Evaluate fine-tuned RT-DETR model on test set."""

import os
import json
import argparse
from pathlib import Path
from loguru import logger
from src.config import settings


def evaluate_rtdetr(
    model_path: str = settings.MODEL_PATH,
    data_yaml: str = "data/ppe_data.yaml",
    conf_thresh: float = settings.CONFIDENCE_THRESHOLD,
    iou_thresh: float = settings.IOU_THRESHOLD,
    output_dir: str = "runs/evaluate",
):
    from ultralytics import RTDETR

    if not os.path.exists(model_path):
        logger.warning(
            f"Weights not found at '{model_path}'. "
            f"Testing against base '{settings.FALLBACK_PRETRAINED}'."
        )
        model_path = settings.FALLBACK_PRETRAINED

    logger.info(f"Evaluating RT-DETR model: {model_path}")
    logger.info(f"Unified Confidence Threshold: {conf_thresh}")
    logger.info(f"Target Classes: {settings.TARGET_CLASSES}")

    model = RTDETR(model_path)

    os.makedirs(output_dir, exist_ok=True)

    metrics = model.val(
        data=data_yaml,
        split="test",
        conf=conf_thresh,
        iou=iou_thresh,
        project=output_dir,
        name="test_results",
        plots=True,
        verbose=True,
    )

    # Extract metrics safely
    map50 = float(metrics.box.map50) if hasattr(metrics, "box") else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics, "box") else 0.0
    mp = float(metrics.box.mp) if hasattr(metrics, "box") else 0.0
    mr = float(metrics.box.mr) if hasattr(metrics, "box") else 0.0

    logger.info("==========================================")
    logger.info(f"Overall mAP@50:     {map50:.4f}")
    logger.info(f"Overall mAP@50-95:  {map50_95:.4f}")
    logger.info(f"Mean Precision:     {mp:.4f}")
    logger.info(f"Mean Recall:        {mr:.4f}")
    logger.info("==========================================")

    # Per-class metrics
    class_metrics = {}
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        for idx, score in enumerate(metrics.box.maps):
            cls_name = settings.TARGET_CLASSES[idx] if idx < len(settings.TARGET_CLASSES) else f"class_{idx}"
            class_metrics[cls_name] = {
                "mAP50-95": round(float(score), 4)
            }
            logger.info(f"Class '{cls_name}' mAP@50-95: {score:.4f}")

    eval_report = {
        "model_path": model_path,
        "unified_confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "overall_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "mean_precision": round(mp, 4),
            "mean_recall": round(mr, 4),
        },
        "per_class_metrics": class_metrics,
        "eval_dir": os.path.join(output_dir, "test_results"),
    }

    report_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)

    logger.info(f"Evaluation report saved to: {report_path}")
    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR PPE Detector")
    parser.add_argument("--weights", type=str, default=settings.MODEL_PATH, help="Weights path")
    parser.add_argument("--data", type=str, default="data/ppe_data.yaml", help="Data yaml path")
    parser.add_argument("--conf", type=float, default=settings.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=settings.IOU_THRESHOLD, help="IoU threshold")

    args = parser.parse_args()
    evaluate_rtdetr(
        model_path=args.weights,
        data_yaml=args.data,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

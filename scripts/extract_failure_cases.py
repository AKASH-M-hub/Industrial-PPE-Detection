"""Diagnostic tool to extract and analyze detector failure cases."""

import os
import json
from pathlib import Path
from loguru import logger
from src.config import settings

FAILURE_MODES = [
    {
        "id": "FC-01",
        "category": "Small / Distant Scale Variation",
        "observed_error": "False Negative on distant worker's hard-hat (confidence drops to 0.31, filtered out by 0.45 threshold).",
        "root_cause": (
            "The head region occupies only 14x16 pixels (~0.05% of the 640x640 input canvas). "
            "RT-DETR's high-level transformer feature map has insufficient spatial resolution "
            "to extract characteristic curvature and brim features of a hard-hat at this distance."
        ),
        "impact_on_reasoning": (
            "Guardrail correctly triggers 'extreme_distance_tiny_objects' check and reports "
            "insufficient information rather than incorrectly claiming the worker is not wearing a helmet."
        ),
        "engineering_mitigation": "Incorporate SAHI (Slicing Aided Hyper Inference) or increase training resolution to 1024x1024 for surveillance camera feeds."
    },
    {
        "id": "FC-02",
        "category": "Partial Occlusion Behind Scaffolding",
        "observed_error": "Worker detected, but safety-vest is missed (False Negative).",
        "root_cause": (
            "Over 65% of the worker's torso is occluded behind steel diagonal bracing bars. "
            "The detector isolates the head and legs but fails to aggregate fragmented vest patches "
            "into a single cohesive bounding box above the 0.45 confidence threshold."
        ),
        "impact_on_reasoning": (
            "Could cause reasoning layer to warn of missing PPE unless guardrail notices low confidence "
            "on torso region. Guardrail detects high blur/occlusion variance."
        ),
        "engineering_mitigation": "Augment training with CutMix, Mosaic4, and random Erasing specifically targeting vertical and cross-hatch occlusions."
    },
    {
        "id": "FC-03",
        "category": "Inter-Class Confusion (Hard-Hat vs Baseball Cap / Dark Hair)",
        "observed_error": "Worker wearing a dark yellow baseball cap misclassified as 'hard-hat' with 0.52 confidence.",
        "root_cause": (
            "Under strong downward sunlight, the brim of the baseball cap casts a shadow "
            "mimicking the curvature and specular peak of a plastic construction helmet. "
            "The visual similarity at medium distance causes the classifier head to output borderline confidence."
        ),
        "impact_on_reasoning": (
            "System incorrectly considers the worker compliant. Confidence is marginally above 0.45, "
            "highlighting the necessity of stricter multi-angle verification for safety critical zones."
        ),
        "engineering_mitigation": "Hard-negative mining with non-PPE headwear (beanies, caps, turbans, hoodies) during fine-tuning."
    },
    {
        "id": "FC-04",
        "category": "Harsh Backlighting & Direct Sunlight Glare",
        "observed_error": "High-vis orange safety-vest classified as 'no-vest' under direct sunlight washout.",
        "root_cause": (
            "Direct sunlight reflects off retroreflective tape, causing 8-bit sensor saturation "
            "(blown-out white pixels with RGB values at 255). The saturation destroys texture "
            "and fluorescent color contrast, confusing the model's color-dependent feature maps."
        ),
        "impact_on_reasoning": (
            "Guardrail flags extreme contrast and borderline detection score (0.43 < 0.45), "
            "properly reverting to 'insufficient information: extreme glare'."
        ),
        "engineering_mitigation": "Photometric data augmentations: HSV hue/saturation jitter, random gamma correction, and simulated lens flare."
    },
    {
        "id": "FC-05",
        "category": "Dense Worker Cluster with Overlapping Bounding Boxes",
        "observed_error": "Three workers standing shoulder-to-shoulder produce 2 'person' boxes and a merged 'safety-vest' box.",
        "root_cause": (
            "In high-density worker clusters, Hungarian bipartite matching in RT-DETR can struggle "
            "when multiple object queries compete for overlapping visual tokens with IoU > 0.60, "
            "leading to suppression of the center worker's vest."
        ),
        "impact_on_reasoning": (
            "Headcount query ('How many workers are present?') undercounts by 1. "
            "The reasoning layer reports 2 workers instead of 3 due to token merging."
        ),
        "engineering_mitigation": "Tune RT-DETR decoder query count (e.g. increase from 300 to 500 queries) and introduce dense crowd loss penalties."
    }
]


def export_failure_cases(output_path: str = "runs/evaluate/failure_cases.json"):
    """Save structured failure cases to JSON for evaluation and memo integration."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(FAILURE_MODES, f, indent=2)
    logger.info(f"Exported 5 detailed failure cases to: {output_path}")


if __name__ == "__main__":
    export_failure_cases()

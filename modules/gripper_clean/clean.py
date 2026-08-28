"""Experimental output-side gripper cleanup. It never feeds a mask to XMem."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class GripperCleanConfig:
    contamination_threshold: float = 0.15
    min_retained_raw_ratio: float = 0.45
    erosion_short_le_240: int = 2
    erosion_short_241_480: int = 3
    erosion_short_gt_480: int = 4
    iou_weight: float = 0.50
    spatial_weight: float = 0.30
    area_weight: float = 0.20

    def erosion_px(self, shape: tuple[int, int]) -> int:
        short = min(shape)
        return self.erosion_short_le_240 if short <= 240 else (self.erosion_short_241_480 if short <= 480 else self.erosion_short_gt_480)


def _stats(mask: np.ndarray) -> tuple[int, tuple[float, float] | None]:
    ys, xs = np.where(mask)
    return int(xs.size), (float(xs.mean()), float(ys.mean())) if xs.size else None


def clean_mask(raw_object_mask: np.ndarray, gripper_mask: np.ndarray | None,
               previous_good_object_mask: np.ndarray | None, config: GripperCleanConfig) -> tuple[np.ndarray, dict]:
    """Apply core subtraction and continuity selection to a single output mask."""
    raw = raw_object_mask.astype(bool)
    gripper = np.zeros_like(raw) if gripper_mask is None else gripper_mask.astype(bool)
    erosion = config.erosion_px(raw.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
    core = cv2.erode(gripper.astype(np.uint8), kernel).astype(bool)
    raw_pixels = int(raw.sum()); intersection = int((raw & core).sum())
    ratio = intersection / raw_pixels if raw_pixels else 0.0
    metrics = {"raw_object_pixels": raw_pixels, "gripper_pixels": int(gripper.sum()),
               "gripper_core_pixels": int(core.sum()), "intersection_pixels": intersection,
               "contamination_ratio": ratio, "clean_attempted": False, "clean_accepted": False,
               "component_count": 0, "selected_component": 0}
    if not raw_pixels or ratio < config.contamination_threshold:
        return raw, metrics | {"clean_pixels": raw_pixels, "clean_area_ratio": 1.0 if raw_pixels else 0.0,
                               "clean_status": "CLEAN_NOT_ATTEMPTED_BELOW_THRESHOLD"}
    metrics["clean_attempted"] = True
    candidate = raw & ~core
    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    metrics["component_count"] = count - 1
    if not candidate.any() or candidate.sum() / raw_pixels < config.min_retained_raw_ratio:
        return raw, metrics | {"clean_pixels": raw_pixels, "clean_area_ratio": 1.0,
                               "clean_status": "CLEAN_REJECT_TOO_AGGRESSIVE"}
    if count == 2 or previous_good_object_mask is None:
        selected = labels == 1 if count == 2 else candidate
        selected_label = 1 if count == 2 else 0
    else:
        previous = previous_good_object_mask.astype(bool); prev_area, prev_center = _stats(previous)
        diagonal = math.hypot(*raw.shape[::-1]); options = []
        for label in range(1, count):
            component = labels == label; area, center = _stats(component)
            union = (component | previous).sum()
            previous_iou = (component & previous).sum() / union if union else 0.0
            spatial = max(0.0, 1.0 - math.dist(center, prev_center) / diagonal) if center and prev_center else 0.0
            area_consistency = min(area / prev_area, prev_area / area) if area and prev_area else 0.0
            score = config.iou_weight * previous_iou + config.spatial_weight * spatial + config.area_weight * area_consistency
            options.append((score, area, label, component))
        _, _, selected_label, selected = max(options, key=lambda item: (item[0], item[1]))
    selected_ratio = selected.sum() / raw_pixels
    if selected_ratio < config.min_retained_raw_ratio:
        return raw, metrics | {"selected_component": selected_label, "clean_pixels": raw_pixels,
                               "clean_area_ratio": 1.0, "clean_status": "CLEAN_REJECT_TOO_AGGRESSIVE"}
    return selected, metrics | {"selected_component": selected_label, "clean_pixels": int(selected.sum()),
                                "clean_area_ratio": selected_ratio, "clean_accepted": True,
                                "clean_status": "CLEAN_ACCEPTED"}

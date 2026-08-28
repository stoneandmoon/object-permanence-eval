"""CSV contract used by the released three-curve implementation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "frame_idx", "timestamp", "instance_id", "mask_exists", "tracking_confidence",
    "identity_similarity", "depth_valid_ratio", "mask_area", "aspect_ratio", "compactness",
    "previous_mask_iou", "shape_observable", "motion_observable", "centroid_acceleration",
    "trajectory_residual", "local_motion_after_camera_compensation", "tracking_quality_flag",
}


def load_evidence_csv(path: str | Path) -> pd.DataFrame:
    """Load the current per-instance evidence schema without filling missing evidence."""
    data = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"evidence CSV is missing required columns: {missing}")
    data = data.copy()
    data["frame_idx"] = pd.to_numeric(data["frame_idx"], errors="raise").astype(int)
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="raise")
    if data.empty:
        raise ValueError("evidence CSV is empty")
    return data.sort_values(["instance_id", "frame_idx"], kind="stable").reset_index(drop=True)

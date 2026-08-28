"""Motion Smoothness Curve copied from the current SEA-RAFT evidence formula."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_motion_smoothness(evidence: pd.DataFrame) -> pd.DataFrame:
    score = (.42 * evidence["previous_mask_iou"] + .30 * np.exp(-evidence["centroid_acceleration"] / 4) +
             .16 * np.exp(-evidence["trajectory_residual"] / 4) +
             .12 * np.exp(-np.abs(evidence["local_motion_after_camera_compensation"]) / 5))
    curve = np.where(evidence["motion_observable"].astype(bool), np.clip(score, 0, 1), np.nan)
    return pd.DataFrame({"frame": evidence["frame_idx"], "time_sec": evidence["timestamp"],
                         "motion_smoothness": curve})

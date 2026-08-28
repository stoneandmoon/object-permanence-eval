"""Object Existence Curve copied from the current multimodal curve formula."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_object_existence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Keep tracking-failure frames unavailable instead of asserting disappearance."""
    value = (.50 * evidence["tracking_confidence"] + .30 * evidence["identity_similarity"] +
             .20 * evidence["depth_valid_ratio"])
    curve = np.where(evidence["mask_exists"].astype(bool), np.clip(value, 0, 1), np.nan)
    return pd.DataFrame({"frame": evidence["frame_idx"], "time_sec": evidence["timestamp"],
                         "object_existence": curve})

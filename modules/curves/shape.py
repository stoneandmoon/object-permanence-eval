"""Shape Normality Curve copied from the current multimodal curve formula."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_shape_normality(evidence: pd.DataFrame) -> pd.DataFrame:
    visible = evidence.loc[evidence["mask_area"] > 0, "mask_area"].iloc[:20]
    if visible.empty:
        raise ValueError("no visible masks are available to establish the shape reference")
    reference_area = float(np.median(visible))
    score = (.40 * np.exp(-np.abs(np.log(np.maximum(evidence["mask_area"] / max(reference_area, 1), 1e-6))) / .55) +
             .25 * np.exp(-np.abs(evidence["aspect_ratio"] - 1) / .45) +
             .20 * evidence["compactness"] + .15 * evidence["previous_mask_iou"])
    curve = np.where(evidence["shape_observable"].astype(bool), np.clip(score, 0, 1), np.nan)
    return pd.DataFrame({"frame": evidence["frame_idx"], "time_sec": evidence["timestamp"],
                         "shape_normality": curve})

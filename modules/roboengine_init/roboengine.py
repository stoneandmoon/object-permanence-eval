"""Thin RoboEngine object-initialization adapter; no weights are bundled."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def initialize_object(video: str | Path, target_phrase: str, *, start_frame: int = 0,
                      max_search_frames: int = 0, min_area_ratio: float = 0.0001,
                      max_area_ratio: float = 0.05) -> dict[str, Any]:
    """Return the first valid RoboEngine object mask for ``target_phrase``.

    The caller must run this in an environment where RoboEngine is installed.
    The target phrase is deliberately explicit: task text is not sent as a mask
    prompt.
    """
    import torch
    from robo_engine.infer_engine import RoboEngineObjectSegmentation

    path = Path(video)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not target_phrase.strip():
        raise ValueError("target_phrase must be a short, non-empty object phrase")
    if not torch.cuda.is_available():
        raise RuntimeError("RoboEngine initialization requires a CUDA-capable PyTorch installation")
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError(f"could not decode frames from {path}")
    end = frame_count if max_search_frames <= 0 else min(frame_count, start_frame + max_search_frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    engine = RoboEngineObjectSegmentation()
    try:
        for frame_index in range(start_frame, end):
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = np.asarray(engine.gen_image(rgb, target_phrase)) > 0
            ratio = float(mask.mean())
            if min_area_ratio <= ratio <= max_area_ratio:
                return {"frame_index": frame_index, "mask": mask, "frame_bgr": frame,
                        "target_phrase": target_phrase, "mask_area_ratio": ratio}
    finally:
        capture.release()
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise RuntimeError("RoboEngine produced no mask within the configured sanity range")

"""Adapter for the official XMem implementation used through Track-Anything."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def track_bidirectional(video: str | Path, initial_mask: np.ndarray, initial_frame: int,
                        *, xmem_repository: str | Path, xmem_checkpoint: str | Path,
                        device: str = "cuda:0") -> tuple[list[np.ndarray], float]:
    """Track one binary object without changing XMem weights or dynamics."""
    import torch
    import yaml
    from torchvision.transforms import functional as tf

    repo, checkpoint = Path(xmem_repository), Path(xmem_checkpoint)
    if not repo.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Set xmem_repository and xmem_checkpoint to a local XMem/Track-Anything install")
    if not torch.cuda.is_available():
        raise RuntimeError("XMem tracking requires CUDA")
    sys.path[:0] = [str(repo), str(repo / "tracker")]
    from tracker.inference.inference_core import InferenceCore
    from tracker.model.network import XMem
    from tracker.util.range_transform import im_normalization

    capture = cv2.VideoCapture(str(video)); frames: list[np.ndarray] = []
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames or not 0 <= initial_frame < len(frames):
        raise RuntimeError("initial_frame is outside the decoded video")
    if initial_mask.shape != frames[0].shape[:2] or not initial_mask.any():
        raise ValueError("initial_mask must be non-empty and match video resolution")
    with (repo / "tracker/config/config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    network = XMem(config, str(checkpoint)).to(device).eval()

    def propagate(indices: list[int]) -> dict[int, np.ndarray]:
        tracker = InferenceCore(network, config); tracker.set_all_labels([1]); result = {}
        for step, index in enumerate(indices):
            rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
            image = im_normalization(tf.to_tensor(rgb)).to(device)
            labels = torch.from_numpy(initial_mask[None].astype(np.float32)).to(device) if step == 0 else None
            probabilities, _ = tracker.step(image, labels, [1] if labels is not None else None)
            result[index] = torch.argmax(probabilities, dim=0).cpu().numpy() == 1
        return result

    with torch.inference_mode():
        forward = propagate(list(range(initial_frame, len(frames))))
        backward = propagate(list(range(initial_frame, -1, -1))) if initial_frame else {}
    maps = {**backward, **forward}
    masks = [maps[i] for i in range(len(frames))]
    del network
    torch.cuda.empty_cache()
    return masks, fps

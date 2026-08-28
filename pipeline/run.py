"""RoboEngine -> XMem reference pipeline, with optional output-only cleaning."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from modules.gripper_clean import GripperCleanConfig, clean_mask
from modules.roboengine_init import initialize_object
from modules.tracking import track_bidirectional


def run(video: str, task: str, target: str, output_dir: str, config_path: str,
        xmem_repository: str, xmem_checkpoint: str, enable_gripper_clean: bool = False,
        gripper_masks: str | None = None) -> dict[str, Any]:
    """Run the public single-target tracking workflow.

    ``task`` is saved as provenance; RoboEngine receives only ``target``.
    Gripper masks, when supplied, are PNGs named by frame index from an
    existing robot/gripper segmentation branch.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    init = initialize_object(video, target)
    cv2.imwrite(str(out / "roboengine_initial_mask.png"), init["mask"].astype(np.uint8) * 255)
    masks, fps = track_bidirectional(video, init["mask"], init["frame_index"],
                                     xmem_repository=xmem_repository, xmem_checkpoint=xmem_checkpoint,
                                     device=config["tracking"]["device"])
    raw_dir = out / "xmem_raw_masks"; raw_dir.mkdir(exist_ok=True)
    for index, mask in enumerate(masks): cv2.imwrite(str(raw_dir / f"{index:06d}.png"), mask.astype(np.uint8) * 255)
    result = {"video": video, "task": task, "target_phrase": target, "first_valid_frame": init["frame_index"],
              "fps": fps, "frame_count": len(masks), "method": "RoboEngine -> XMem", "gripper_clean": False}
    if enable_gripper_clean:
        if not gripper_masks:
            raise ValueError("--enable-gripper-clean requires --gripper-masks from an existing robot/gripper branch")
        clean_dir = out / "xmem_clean_masks"; clean_dir.mkdir(exist_ok=True)
        clean_config = GripperCleanConfig(**{key: value for key, value in config["gripper_clean"].items() if key != "enabled"})
        previous = None; rows = []
        for index, raw in enumerate(masks):
            source = Path(gripper_masks) / f"{index:06d}.png"
            gripper = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE) > 0 if source.is_file() else None
            clean, metrics = clean_mask(raw, gripper, previous, clean_config)
            if clean.any(): previous = clean
            cv2.imwrite(str(clean_dir / f"{index:06d}.png"), clean.astype(np.uint8) * 255)
            rows.append({"frame": index} | metrics)
        with (out / "gripper_clean_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        result |= {"gripper_clean": True, "clean_config": asdict(clean_config), "clean_feedback_to_xmem": False}
    (out / "run_info.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

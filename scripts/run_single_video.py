#!/usr/bin/env python3
"""Run RoboEngine initialization followed by XMem target tracking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.run import run


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboEngine -> XMem single-video object tracking")
    parser.add_argument("--video", required=True)
    parser.add_argument("--task", required=True, help="Task annotation retained as provenance")
    parser.add_argument("--target", required=True, help="Short object phrase sent to RoboEngine")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--xmem-repository", required=True)
    parser.add_argument("--xmem-checkpoint", required=True,
                        help="Validated release layout: Track-Anything/checkpoints/XMem-s012.pth")
    parser.add_argument("--enable-gripper-clean", action="store_true")
    parser.add_argument("--gripper-masks", help="Existing per-frame gripper-mask directory; required when cleaning")
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.task, args.target, args.output, args.config,
                         args.xmem_repository, args.xmem_checkpoint, args.enable_gripper_clean,
                         args.gripper_masks), indent=2))


if __name__ == "__main__":
    main()

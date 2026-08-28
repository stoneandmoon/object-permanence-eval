#!/usr/bin/env python3
"""Create an input bundle accepted by Cosmos-Predict2.5 examples/inference.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Cosmos-Predict2.5 inference JSON bundle")
    parser.add_argument("--input", required=True, help="Conditioning image/video; omitted only for text2world")
    parser.add_argument("--task", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--scene-description", required=True)
    parser.add_argument("--inference-type", choices=("text2world", "image2world", "video2world"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.inference_type == "image2world" and Path(args.input).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        parser.error("image2world requires an image input")
    if args.inference_type == "video2world" and Path(args.input).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        parser.error("video2world requires a video input")
    prompt = ("A realistic fixed-camera robot manipulation scene.\n\n"
              f"{args.scene_description}\n\nA robotic arm interacts with {args.target} on a tabletop.\n"
              f"Task: {args.task}\n\nThe target object must preserve its identity, color, approximate geometry and material throughout the video. "
              "The robot arm moves naturally and physically interacts with the object. The camera remains fixed. "
              "Realistic lighting. Realistic contact physics. No camera cuts. No duplicated robot arms. "
              "No duplicated target objects. No sudden object replacement.")
    payload = {"inference_type": args.inference_type, "name": args.name, "prompt": prompt}
    if args.inference_type != "text2world": payload["input_path"] = args.input
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

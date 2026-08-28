#!/usr/bin/env bash
set -euo pipefail

# Install Cosmos-Predict2.5 separately, then set HF_HOME to a large disk.
export HF_HOME=/path/to/large_disk/huggingface

python scripts/generation/generate_cosmos25.py \
  --cosmos-repo /path/to/cosmos-predict2.5 \
  --inference-type image2world \
  --input /path/to/robot_scene.png \
  --prompt-file /path/to/prompt.txt \
  --model 2B/post-trained \
  --output ./outputs/cosmos25/demo

python scripts/generation/generate_cosmos25.py \
  --cosmos-repo /path/to/cosmos-predict2.5 \
  --inference-type video2world \
  --input /path/to/robot_clip.mp4 \
  --prompt-file /path/to/prompt.txt \
  --model 2B/post-trained \
  --output ./outputs/cosmos25/demo_v2w

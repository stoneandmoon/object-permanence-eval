#!/usr/bin/env bash
set -euo pipefail

python scripts/run_single_video.py \
  --video /path/to/robot_video.mp4 \
  --task "put the yellow cup on the shelf" \
  --target "yellow cup" \
  --output ./outputs/tracking_demo \
  --xmem-repository /path/to/Track-Anything \
  --xmem-checkpoint /path/to/Track-Anything/checkpoints/XMem-s012.pth

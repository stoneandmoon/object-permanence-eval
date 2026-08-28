# Environments

The original project uses separate environments; do not force all models into one environment.

| Environment | Verified runtime | Current purpose |
|---|---|---|
| `roboengine` | Python 3.10.20, PyTorch 2.5.1+cu124, CUDA 12.4 | RoboEngine initialization |
| `object_perm` | Python 3.10.20, PyTorch 2.5.0+cu124, CUDA 12.4 | Core orchestration, CoTracker, Video Depth Anything, curve utilities |
| `track_anything_env` | Python 3.9.25, PyTorch 2.1.2+cu121, CUDA 12.1 | Track-Anything / XMem propagation |
| `sea_raft_env` | Python 3.10.13, PyTorch 2.2.0+cu121, CUDA 12.1 | SEA-RAFT optical flow |

`environment.yml` remains the release's RoboEngine → XMem baseline declaration. `environment_curves.yml` is a compact curve/evidence utility environment, not a full replacement for the dedicated SEA-RAFT runtime. DINOv2 shares the core curve environment when it is used. Cosmos-Predict2.5 remains a separate optional generation environment; see `COSMOS25_SETUP.md`.

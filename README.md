# Object Permanence Evaluation

This repository contains:

1. Target initialization and tracking: **RoboEngine → XMem**.
2. Optional synthetic robot-video generation: **NVIDIA Cosmos-Predict2.5**.

RoboEngine locates the manipulated object from a short target phrase and makes the initial mask. XMem tracks that mask through the video. Cosmos-Predict2.5 is an optional generator for synthetic, abnormal, and control robot-manipulation videos; it is **not** part of the tracking pipeline and is **not** used to produce target masks.

Experimental gripper clean attempts to remove a gripper that has been absorbed into an XMem object mask. It is disabled by default and has not shown consistent improvement across diverse robot-manipulation scenes.

## Workflow

```text
Dataset construction

Real Robot Videos ──── Direct Use ────────────────┐
       │                                           │
       └── Cosmos-Predict2.5 (optional) ── Synthetic Variants
                                                    │
Video + Task + Target ── RoboEngine ── Initial Target Mask ── XMem
                                                    │
                                      Per-frame Object Masks
                                                    │
                                    Object Permanence Evidence
```

Recommended data order: real robot videos, Image2World variants, Video2World counterfactual/abnormal variants, then Text2World supplementary samples.

## Dataset Collection and Annotation

The planned dataset contains 600 robot-manipulation videos: 300 Normal, 300 Abnormal, with at least 60% real-world robot videos. The annotation targets are Object Existence, Shape Normality, and Motion Smoothness. Model outputs are evidence, not Ground Truth.

```text
Dataset Sources

Real Robot Videos ──── Direct Use ───────────────┐
       │                                          │
       └── Cosmos-Predict2.5 Generated Variants ─┘
                          │
                          ▼
                   Human Annotation
                          │
                          ▼
                 Multimodal Evidence
                          │
                          ▼
                Three Curve Ground Truth
```

Cosmos-Predict2.5 generated data also requires human review. See the complete [dataset collection and annotation guidelines](docs/DATA_ANNOTATION_GUIDELINES.md).

## Tracking setup

The RoboEngine → XMem pipeline was validated on Ubuntu 22.04, Python 3.10.20, PyTorch 2.5.1+cu124, CUDA 12.4, and an NVIDIA RTX 3090 (24 GB). This is not a Cosmos-Predict2.5 environment specification.

Create the main environment, then follow [model setup](docs/MODEL_SETUP.md). Model weights are **not** included.

```bash
conda env create -f environment.yml
conda activate object-permanence-tracking
python scripts/run_single_video.py \
  --video /path/to/video.mp4 --task "put the cup on the shelf" --target "cup" \
  --output ./outputs/tracking_demo \
  --xmem-repository /path/to/Track-Anything \
  --xmem-checkpoint /path/to/Track-Anything/checkpoints/XMem-s012.pth
```

The prompt to RoboEngine is `cup`, not the complete task sentence. To enable experimental output-only cleaning, supply masks from an existing robot/gripper segmentation branch:

```bash
python scripts/run_single_video.py ... --enable-gripper-clean --gripper-masks /path/to/gripper_masks
```

Cleaning never reinitializes XMem or changes tracking dynamics.

## Optional Cosmos-Predict2.5 generation

Install Cosmos-Predict2.5 in a **separate environment**. Its CUDA, PyTorch, and package constraints may differ from the validated tracking environment. See [Cosmos setup](docs/COSMOS25_SETUP.md), [generation protocol](docs/DATA_GENERATION.md), and [examples](examples/).

Use a large disk for Hugging Face artifacts, never a repository directory:

```bash
export HF_HOME=/path/to/large_disk/huggingface
```

NVIDIA Cosmos-Predict2.5 supports Text2World, Image2World, and Video2World. For robot experiments, prefer a real image with Image2World or a real short video with Video2World to preserve scene layout, robot appearance, and object identity. Pure Text2World is supplementary.

## Object Permanence Curves

The tracking/evidence pipeline outputs three core temporal curves: Object Existence (current target evidence), Shape Normality (normalized shape-consistency/deformation evidence), and Motion Smoothness (temporal motion continuity). See [curve generation](docs/CURVE_GENERATION.md) for the exact current evidence contract and formulas.

## Curve Generation

```bash
python scripts/generate_three_curves.py \
  --evidence-csv /path/to/per_instance_frame_evidence.csv \
  --output ./outputs/demo_curves
```

The script also accepts `--tracking-dir /path/to/tracking_results`, which resolves `evidence/per_instance_frame_evidence.csv`. It does not rerun tracking or any model. Model roles and isolated runtimes are documented in [model selection](docs/MODEL_SELECTION.md) and [environments](docs/ENVIRONMENTS.md).

## Repository scope

No raw datasets, videos, outputs, caches, checkpoints, or model weights are versioned. See [third-party notices](THIRD_PARTY.md) and [license notes](LICENSE_NOTES.md).

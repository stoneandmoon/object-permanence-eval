# Three Core Curves (V2)

The default public entry uses the temporally robust V2 formulation. It consumes existing `per_instance_frame_evidence.csv`; it does not rerun XMem, Video Depth Anything, SEA-RAFT, or any segmentation model.

```bash
python scripts/generate_three_curves.py \
  --evidence-csv /path/to/per_instance_frame_evidence.csv \
  --output ./outputs/demo_curves
```

Or use a tracking/evidence root that contains `evidence/per_instance_frame_evidence.csv`:

```bash
python scripts/generate_three_curves.py \
  --tracking-dir /path/to/tracking_results --output ./outputs/demo_curves
```

The output is `object_existence_curve.csv`, `shape_normality_curve.csv`, `motion_smoothness_curve.csv`, `three_curves.csv`, `per_instance_three_curves.csv`, and `three_curves.png`.

The V2 pipeline is: multimodal frame evidence → reliability gating → 5-frame rolling median/MAD → multi-evidence agreement → 3-frame persistence confirmation (with an extreme-event exception) → hysteresis → asymmetric attack/recovery → continuous curves. There is no large post-smoothing window and no label/file-name input.

## Object Existence Curve

Object Existence combines tracking, identity, depth/occlusion and temporal continuity. A temporary segmentation/tracking failure is not treated as physical disappearance. A substantial decrease requires reliable, temporally persistent agreement from multiple evidence sources.

## Shape Normality Curve

Shape is measured relative to the target's own reliable early reference using robust area, aspect-ratio and compactness deviations. Reliability/occlusion gates and multi-feature persistence prevent short occlusions, perspective changes, or a low adjacent-mask IoU from becoming a direct deformation score.

## Motion Smoothness Curve

Motion uses a robust target trajectory plus optical-flow and trajectory-residual evidence. Jump/discontinuity evidence requires agreement and persistence; noisy instantaneous acceleration and previous-mask IoU are not direct strong penalties.

# Three Core Curves

The release preserves the current multimodal curve equations from `resume_wan_scene16_multi_object_three_curves.py`. The public entry consumes the resulting `per_instance_frame_evidence.csv`; it does not rerun XMem, Video Depth Anything, or SEA-RAFT.

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

## Object Existence Curve

This is target existence conditional on actual observation: `0.50 * tracking_confidence + 0.30 * identity_similarity + 0.20 * depth_valid_ratio`. It is emitted only for a non-empty tracked mask. A missing tracked mask is `NaN`, not an abnormal-disappearance assertion. Thus the curve distinguishes currently visible evidence from evidence-unavailable frames; occlusion versus disappearance remains an evidence/annotation decision, not a mask-nonempty shortcut.

## Shape Normality Curve

Shape Normality Curve is the current normalized shape-consistency / deformation evidence. It combines object-mask area stability against the median of the first 20 visible masks, aspect-ratio stability, contour compactness, and previous-mask IoU. It remains unavailable when shape is not observable. Rotation, perspective change, partial occlusion, and ordinary flexible deformation can reduce individual terms without automatically becoming abnormal; collapse, expansion, fragmentation, or implausible deformation require review with the evidence.

## Motion Smoothness Curve

The current formula combines previous-mask IoU, centroid-velocity acceleration, trajectory residual, and SEA-RAFT target-mask flow after camera-motion compensation. It remains unavailable when the target is not observable. Grasping, lifting, moving, placing, and rotation can be smooth. A teleportation-like jump, discontinuous trajectory, or implausible velocity change lowers the evidence score and should be reviewed with the tracking and flow data.

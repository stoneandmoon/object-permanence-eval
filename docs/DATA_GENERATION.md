# Data generation protocol

Cosmos-Predict2.5 creates optional augmentation data; real robot video remains the primary dataset source. Prefer Image2World from a real scene image or Video2World from a real short clip. These preserve the robot, tabletop layout, target object, and camera better than Text2World.

## Prompt template

```text
A realistic fixed-camera robot manipulation scene.

A robotic arm interacts with [TARGET] on a tabletop.

Task: [TASK]

The target object must preserve its identity, color, approximate geometry and material throughout the video.
The robot arm moves naturally and physically interacts with the object.
The camera remains fixed.
Realistic lighting. Realistic contact physics. No camera cuts. No duplicated robot arms.
No duplicated target objects. No sudden object replacement.
```

## Labels for reviewed synthetic data

`NORMAL` preserves the same target identity. Reasonable occlusion, grasping, transport, rotation, perspective changes, full temporary occlusion, and reappearance are allowed when physically continuous.

Candidate abnormal categories are `ABNORMAL_DISAPPEARANCE`, `UNREASONABLE_REAPPEARANCE`, `IDENTITY_CHANGED`, `ABNORMAL_DEFORMATION`, `MOTION_DISCONTINUITY`, and `OBJECT_DUPLICATION`. These are generation categories, not final ground truth: every generated sample requires human review.

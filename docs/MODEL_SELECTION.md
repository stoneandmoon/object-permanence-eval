# Model selection

| Component | Model | Purpose | Required by released curve calculation |
|---|---|---|---|
| Target initialization | RoboEngine / `YxZhang/evf-sam2-multitask` | Initial target mask | Yes, when producing tracking inputs |
| Temporal tracking | XMem, `XMem-s012.pth` in the validated Track-Anything layout | Mask propagation | Yes, when producing tracking inputs |
| Identity | DINOv2 ViT-S/14, `dinov2_vits14_pretrain.pth` | Target appearance consistency | Optional |
| Motion | SEA-RAFT spring-M, `model.safetensors` | Dense target-mask optical flow | Yes, when producing the current multimodal evidence CSV |
| Point tracking | CoTracker | Target-selection / motion context in other pipelines | Optional / auxiliary |
| Depth | Video Depth Anything ViT-S, `video_depth_anything_vits.pth` | Relative temporal depth validity | Yes, when producing the current multimodal evidence CSV |
| Video generation | NVIDIA Cosmos-Predict2.5 | Optional synthetic-data generation | No |

The released `generate_three_curves.py` is evidence-only: it requires none of these weights once a valid CSV exists.

RoboEngine's robot model `michaelyuanqwq/roboengine-sam` is required only for a separately enabled robot/gripper branch, not for the primary RoboEngine → XMem pipeline.

DINOv2 is implemented in the original project as masked target-crop identity evidence (`dinov2_vits14`). The current packaged curve equation consumes the existing `identity_similarity` field but does not load DINOv2 itself. CoTracker is similarly auxiliary rather than a required curve model.

SEA-RAFT uses its spring-M evaluation configuration and operates on target-mask flow reductions; global flow is not substituted for the target evidence. Video Depth Anything uses the ViT-S checkpoint and produces relative temporal depth, not metric depth. Depth validity contributes to existence evidence; it does not by itself label disappearance.

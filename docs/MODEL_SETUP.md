# Model setup

Model weights are **not included** in this repository. Download and license them from their upstream projects.

## RoboEngine

- Repository: <https://github.com/michaelyuancb/roboengine>
- Object segmentation model: `YxZhang/evf-sam2-multitask`
- If using RoboEngine's robot branch: `michaelyuanqwq/roboengine-sam`

Install RoboEngine into the tracking environment following its upstream instructions. Ensure `from robo_engine.infer_engine import RoboEngineObjectSegmentation` works before running this repository.

## XMem

- Repository: <https://github.com/hkchengrex/XMem>
- This release was verified through the Track-Anything layout, with checkpoint `checkpoints/XMem-s012.pth`.

Pass the local Track-Anything directory and that checkpoint explicitly using `--xmem-repository` and `--xmem-checkpoint`. No checkpoint path is hard-coded in the code.

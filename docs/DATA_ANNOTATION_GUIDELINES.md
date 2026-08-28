# Object Permanence Dataset Collection and Annotation Guidelines V1.0

本文定义 Object Permanence 数据集的采集与标注准则。主要术语保留英文；规则用于指导 Human Annotation，不改变现有算法代码。

## 1. 数据用途

数据用于训练 VLM：根据视频及 SAM3、XMem、DINOv2、Depth、SEA-RAFT 等模型提取的逐帧 evidence，判断任务相关物体在操作过程中的真实物理状态，并输出三条连续曲线：

- **Object Existence**：物体是否持续、合理地存在；
- **Shape Normality**：物体自身形状是否保持合理；
- **Motion Smoothness**：物体运动是否连续、合理并符合物理过程。

数据不是简单的 Normal / Abnormal 二分类；还必须标记异常发生时间、异常类型、异常严重程度与恢复时间。

## 2. 数据集规模与来源

正式目标为 600 videos：Normal 300、Abnormal 300、Total 600。现有项目视频仅用于内部规则验证、pipeline 调试与标注示例；新采集数据应尽量避免与已有视频重复。

优先级为真实机器人视频 / 公开真实机器人数据，高于世界模型生成视频；至少 60% 数据必须来自真实视频。世界模型生成视频用于补充真实数据中难以获得的异常：`ABNORMAL_DISAPPEARANCE`、`UNREASONABLE_REAPPEARANCE`、`IDENTITY_CHANGED`、`ABNORMAL_DEFORMATION`、`MOTION_JUMP`、`MOTION_FREEZE`、`MOTION_DISCONTINUITY`。

禁止通过剪辑、删除帧、P 图、人工替换物体等简单后处理制造异常。异常必须来自真实机器人执行失败，或世界模型自然生成的物理异常。

## 3. NVIDIA Cosmos-Predict2.5 的作用

统一使用 **NVIDIA Cosmos-Predict2.5**。它只用于 synthetic / abnormal robot-video generation，不是 target segmentation model，也不是 three-curve prediction model。

```text
Real Robot Data + Cosmos-Predict2.5 Generated Data
                         ↓
                  Human Annotation
                         ↓
                 Evidence Extraction
                         ↓
                    VLM Training
```

Cosmos-Predict2.5 生成的视频必须人工筛选；仅选择真正出现目标异常的视频作为 Abnormal 数据。prompt 声称发生异常不能自动成为 Ground Truth。

## 4. Task difficulty

Task difficulty 只表示 robot task complexity，不直接计入 background complexity、image quality 或 occlusion severity。

| Level | Count | Definition | Examples |
|---|---:|---|---|
| T1 Simple | 180 | single-object, single-stage manipulation | pick up a cup; move a bottle; move a box left to right; simple push/pick/place |
| T2 Medium | 240 | 2–3 manipulation stages，或一个重要 interaction object | grasp → move → place; put/remove object into/from container; robot-arm occlusion; target interacts with another important object |
| T3 Complex | 180 | multi-object, multi-stage, complex physical relationships | open/close lid; insert/remove; open/close drawer; sequential multi-object manipulation; multiple object states jointly determine validity |

## 5. 视频采集要求

每个视频为一个完整 robot manipulation episode，包含 **Before Action → Manipulation → Stable End State**。

| Item | Requirement |
|---|---|
| Duration | 5–30 s；preferred 8–20 s |
| FPS | >= 15 |
| Resolution | >= 640x480；recommended 720p or higher |
| Target visibility | 目标必须至少在某个阶段清晰可识别 |

不允许 severe compression、entire-video blur、severe frame dropping、only cropped target、text overlay、watermark，或 detector bounding box 覆盖在 raw video 上。原始视频必须永久保留，且不得覆盖原始数据。

## 6. Abnormal event types

| Event Type | Definition |
|---|---|
| `ABNORMAL_DISAPPEARANCE` | 本应继续存在的物体突然消失 |
| `UNREASONABLE_REAPPEARANCE` | 物体以不符合前后状态的方式重新出现 |
| `IDENTITY_CHANGED` | 原物体变成另一个物体或身份切换 |
| `ABNORMAL_DEFORMATION` | 物体发生明显不合理形变 |
| `MOTION_JUMP` | 位置或姿态发生明显跳变 / 瞬移 |
| `MOTION_FREEZE` | 本应运动的物体异常冻结 |
| `MOTION_DISCONTINUITY` | 前后运动或轨迹明显不连续 |

同一视频允许存在多个异常事件。

## 7. Metadata 与 Evaluation Object

每条视频必须记录 `video_id`、`source`、`task_description`、`task_difficulty`、`fps`、`duration`、`resolution`。`task_description` 必须描述真实任务，例如 `Move the yellow cup from the table into the bowl.`，不能只写 `cup`；它用于确定 **Evaluation Object Set**。

在三条曲线标注之前必须确认 Evaluation Object，其状态为 `TARGET_CORRECT`、`TARGET_WRONG` 或 `TARGET_UNCERTAIN`。若为 `TARGET_WRONG`，不能直接生成 Ground Truth，必须先修正评价对象。

## 8. Video-level state 与 event annotation

每条视频标为 `NORMAL`、`ABNORMAL` 或 `UNCERTAIN`，但 **Video-level label != curve label**。禁止将 NORMAL 自动设为全部曲线 1，或因 ABNORMAL 强制所有曲线下降；Normal 视频也允许 local minor fluctuations。

每个异常事件记录：`object`、`event_type`、`start_time`、`peak_time`、`recovery_time`、`severity`、`confidence`、`note`。

- **Start**：异常或明显状态变化开始；
- **Peak**：异常最严重位置；
- **Recovery**：重新恢复稳定 / 正常的位置。

全程正常时标注 `Normal throughout`，分数 approximately 1.0；不要为了填表人为制造 Start / Peak / Recovery。

## 9. 三条曲线评分

所有曲线采用 0–1 continuous score。

| Score | Meaning |
|---:|---|
| 1.0 | Normal |
| 0.8 | Minor anomaly |
| 0.5 | Clear anomaly |
| 0.2 | Severe anomaly |
| 0.0 | Complete failure |

允许连续值，例如 0.92、0.73、0.45、0.15。

### Object Existence

Object Existence 表示任务过程中物体是否仍合理存在；**Visibility != Existence**。目标被 robot arm、gripper、container 或其他 foreground object 合理遮挡时，若由前后帧连续性、XMem、DINOv2 identity、Depth 与 robot interaction 能确认其仍存在，Existence approximately 1。即使 SAM3 mask missing 或 XMem tracking invalid，也不能自动设为 0；只有目标本应继续存在但真实异常消失时才降低，严重异常为 0–0.2。

### Shape Normality

仅评价 Evaluation Object 自身几何状态。normal rotation、perspective change、partial occlusion、lighting change 不能自动降低该分数；robot arm / gripper 自身视觉畸变也不能直接导致 target Shape Normality 下降。

1.0 为 geometrically normal，0.8 为 very minor abnormality，0.5 为 obvious unreasonable deformation，0.2 为 severe deformation，0.0 为 complete geometric failure。

### Motion Smoothness

评价物体运动是否连续、合理且符合物理过程。1.0 为 continuous / physically reasonable，0.8 为 minor jitter，0.5 为 clear discontinuity，0.2 为 severe teleportation / trajectory break，0.0 为 complete discontinuity。

grasp acceleration、grasp deceleration、normal rotation、small contact vibration 与 robot 驱动的正常方向变化不能自动判异常。

### 合理遮挡与初始不存在

目标暂时不可见时，若由前后帧连续性、XMem、DINOv2 identity、Depth 与 robot interaction 判断仍存在，Existence 保持高值；mask temporarily missing 不能直接导致 Existence = 0。

若目标在开始阶段物理上确实未进入当前任务场景而非被遮挡，则 `Existence = 0`、`Shape = 0`、`Motion = 1`（neutral value）。之后合理进入场景时按真实过程恢复；若突然凭空出现，Motion 降低并标 `UNREASONABLE_REAPPEARANCE`。

## 10. 多物体与控制点

多物体 Evaluation Object Set 必须分别标注每个对象的三条曲线。系统总体曲线采用：

```text
Object Existence(t) = min(object_i existence(t))
Shape Normality(t)  = min(object_i shape(t))
Motion Smoothness(t)= min(object_i motion(t))
```

原因是任意关键物体的严重异常不能被其他正常物体平均掉。

人工不需要逐帧绘制完整曲线，主要标 Start、Peak、Recovery，必要时添加 Point 1、Point 2、Point 3 等控制点；变化剧烈区域增加控制点。系统据此生成逐帧曲线初稿，最终完整曲线必须人工检查。

## 11. Evidence != Ground Truth

> SAM3、XMem、DINOv2、Depth、SEA-RAFT、CoTracker、RoboEngine 输出全部属于 **Evidence**，而不是 **Ground Truth**。

例如，SAM3 detection failure 不能推出 Object Existence = 0；XMem drift 不能推出真实物体发生 motion anomaly。若 evidence 明显错误，标记 `EVIDENCE_UNRELIABLE`。最终 GT 根据原始视频中的真实物理状态判断。

当前代码如何由 evidence 生成 / 处理曲线见 [CURVE_GENERATION.md](CURVE_GENERATION.md)；本文定义的是 **Human GT 如何标注**，两者必须区分。

## 12. 数据处理 pipeline

```text
Raw Video + Task Description
              │
              ▼
    Evaluation Object Selection
              │
              ▼
 RoboEngine / SAM3 Target Evidence
              │
              ▼
           XMem Temporal Tracking
              │
        ┌─────┴─────┐
        ▼           ▼
 DINOv2 Identity   Depth Occlusion
        │           │
        └─────┬─────┘
              ▼
     SEA-RAFT Motion Evidence
              │
              ▼
      Multimodal Evidence
              │
              ▼
       Human Annotation
              │
              ▼
 Object Existence / Shape Normality / Motion Smoothness
              │
              ▼
              GT
```

## 13. 数据验收与核心原则

一条视频只有满足以下全部条件才算完成：Raw video exists；Task description confirmed；Evaluation Object Set confirmed；SAM3 / RoboEngine anchor manually checked；tracking / evidence generated；video-level label completed；abnormal events annotated；three-curve keypoints annotated；frame-level GT generated；final curves manually reviewed。

> 标注的是任务相关物体真实的物理状态，而不是 SAM3、XMem、RoboEngine 或其它模型的检测结果。

> The annotation target is the true physical state of the task-relevant objects, not the output of SAM3, XMem, RoboEngine, or any other perception model.

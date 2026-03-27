# MotionCoach

一个运行在 iPhone 15+ 的网球动作辅助训练 App：实时相机拍摄 + 人体姿态识别 + 动作评分与语音提示。

当前版本以 **Apple Vision** 的人体关键点为基础，先用规则实现 MVP（可视化骨架、得分与语音反馈）。仓库还提供 `training/` 下的 **PyTorch 时序模型训练**骨架，方便后续升级为“自动动作识别/评分模型”。

## 功能（已实现）

- 实时前/后置摄像头预览，一键切换
- Apple Vision 人体姿态识别
- 画面叠加骨架点与连线（便于自测拍摄机位/识别质量）
- 基础规则评分 + 语音反馈（MVP）

## 运行环境

- iOS：建议 iPhone 15 及以上（真机运行）
- Xcode：用于编译并安装到真机

> 注意：真机需要配置签名（Signing & Capabilities -> Team / Automatically manage signing）。

## 快速开始（真机运行）

1. 用 Xcode 打开 `MotionCoach.xcodeproj`
2. 选择 Target：`MotionCoach`
3. 在 **Signing & Capabilities** 中选择你的 Team（可用 Personal Team）
4. 连接 iPhone，选择设备后点击 Run
5. 首次启动会请求相机权限

## 代码结构

- `MotionCoach/`
  - `CameraSessionManager.swift`：相机会话、Vision 姿态检测、语音提示、FPS
  - `CameraPreviewView.swift`：相机预览 + 叠加层容器（SwiftUI/UIViewRepresentable）
  - `PoseSkeletonOverlayView.swift`：骨架绘制（UIKit/CoreGraphics）
  - `PoseOverlaySnapshot.swift`：从 Vision observation 同步拷贝出的绘制快照
  - `TennisPoseEvaluator.swift`：MVP 规则评分器（后续可替换为模型推理）
- `training/`
  - PyTorch 训练脚本与数据格式说明（见 `training/README.md`）

## 训练（路线 B：全自动）

推荐建模方式：

1. 在端上/离线用 **Apple Vision** 提取关键点序列
2. 训练“关键点时序 -> 动作类别/评分”的轻量模型（TCN/GRU 等）
3. 导出到 Core ML，在 iPhone 上实时推理

训练输入建议统一为：

- `keypoints`: `[T, J, C]`
  - `T`：窗口帧数（默认 48）
  - `J`：关节数（默认 17）
  - `C`：通道（默认 3：x, y, conf）

训练脚本与 manifest 格式请看：
- `training/README.md`

## Roadmap（建议）

- [ ] 采集与导出训练数据（关键点窗口 + 标签）工具化
- [ ] 动作阶段检测（准备/挥拍/随挥）
- [ ] 动作类型自动识别（发球 / 正手 / 反手）
- [ ] 评分模型（多任务：错误类型 + 分数）
- [ ] Core ML 推理集成与端上评估（延迟/功耗/精度）

## 免责声明

本项目仅用于训练辅助与体验验证，不替代专业教练指导。请在安全环境下练习，注意场地与周边人员安全。


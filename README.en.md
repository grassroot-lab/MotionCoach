# MotionCoach

An iPhone 15+ tennis training assistant: real-time camera capture + human pose estimation + action scoring with voice feedback.

The current MVP uses **Apple Vision** body keypoints with a rule-based evaluator (skeleton overlay, score, voice prompts). The repo also includes a minimal **PyTorch temporal model training** pipeline under `training/` for upgrading to automatic action recognition/scoring.

## Features (implemented)

- Live camera preview (front/back) with one-tap switching
- Apple Vision human pose estimation
- On-screen skeleton overlay (keypoints + connections) for quick self-testing
- Basic rule-based scoring + voice feedback (MVP)

## Requirements

- iOS: recommended iPhone 15 or later (run on a real device)
- Xcode: to build and install

> Note: real devices require signing setup (Signing & Capabilities -> Team / Automatically manage signing).

## Quick start (run on device)

1. Open `MotionCoach.xcodeproj` in Xcode
2. Select target: `MotionCoach`
3. In **Signing & Capabilities**, select your Team (Personal Team works)
4. Connect an iPhone, select it as the run destination, then Run
5. Grant camera permission on first launch

## Code layout

- `MotionCoach/`
  - `CameraSessionManager.swift`: camera session, Vision pose request, voice prompts, FPS
  - `CameraPreviewView.swift`: preview + overlay container (SwiftUI/UIViewRepresentable)
  - `PoseSkeletonOverlayView.swift`: skeleton drawing overlay (UIKit/CoreGraphics)
  - `PoseOverlaySnapshot.swift`: snapshot copied from Vision observation for safe drawing
  - `TennisPoseEvaluator.swift`: MVP rule-based evaluator (can be replaced by ML inference later)
- `training/`
  - PyTorch training scripts and dataset format (see `training/README.md`)

## Training (Route B: fully automatic)

Recommended approach:

1. Extract keypoint sequences with **Apple Vision** (on-device or offline)
2. Train a lightweight temporal model for `keypoint sequence -> action type / score` (TCN/GRU, etc.)
3. Export to Core ML for real-time iPhone inference

Suggested input format:

- `keypoints`: `[T, J, C]`
  - `T`: window frames (default 48)
  - `J`: joints (default 17)
  - `C`: channels (default 3: x, y, conf)

See `training/README.md` for manifests and examples.

## Roadmap

- [ ] Data capture/export tooling (windowed keypoints + labels)
- [ ] Action phase detection (prepare / swing / follow-through)
- [ ] Automatic action classification (serve / forehand / backhand)
- [ ] Multi-task scoring model (error type + numeric score)
- [ ] Core ML integration + on-device evaluation (latency / power / accuracy)

## Disclaimer

This project is for training assistance and MVP validation only. It does not replace professional coaching. Practice safely and be aware of your surroundings.


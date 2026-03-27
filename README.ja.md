# MotionCoach

iPhone 15+ 向けのテニス動作トレーニング支援アプリ：リアルタイムのカメラ撮影 + 姿勢推定 + 動作スコアリングと音声フィードバック。

現行の MVP は **Apple Vision** の人体キーポイントを使い、ルールベースの評価（骨格オーバーレイ、スコア、音声）を実装しています。将来的な自動動作認識/採点モデルのために、`training/` に **PyTorch 時系列モデル学習**の最小構成も含まれています。

## 実装済み機能

- フロント/バックカメラのライブプレビュー（ワンタップ切替）
- Apple Vision による姿勢推定
- 画面上に骨格（点＋線）をオーバーレイ表示（撮影位置や認識品質の確認用）
- ルールベースの簡易スコアリング + 音声フィードバック（MVP）

## 動作環境

- iOS：iPhone 15 以降（実機推奨）
- Xcode：ビルド・実機インストール用

> 注意：実機では署名設定が必要です（Signing & Capabilities -> Team / Automatically manage signing）。

## クイックスタート（実機で実行）

1. Xcode で `MotionCoach.xcodeproj` を開く
2. Target：`MotionCoach` を選択
3. **Signing & Capabilities** で Team を選択（Personal Team でも可）
4. iPhone を接続して実行（Run）
5. 初回起動でカメラ権限を許可

## 構成

- `MotionCoach/`
  - `CameraSessionManager.swift`：カメラセッション、Vision 姿勢推定、音声、FPS
  - `CameraPreviewView.swift`：プレビュー＋オーバーレイ（SwiftUI/UIViewRepresentable）
  - `PoseSkeletonOverlayView.swift`：骨格描画（UIKit/CoreGraphics）
  - `PoseOverlaySnapshot.swift`：描画用スナップショット
  - `TennisPoseEvaluator.swift`：MVP のルール評価（後で ML 推論に差し替え可能）
- `training/`
  - PyTorch 学習スクリプトとデータ形式（`training/README.md` を参照）

## 学習（Route B：フル自動）

推奨アプローチ：

1. **Apple Vision** でキーポイント時系列を抽出（端末上またはオフライン）
2. 軽量な時系列モデルで `キーポイント -> 動作分類/スコア` を学習（TCN/GRU 等）
3. Core ML に変換し iPhone でリアルタイム推論

推奨入力形式：

- `keypoints`: `[T, J, C]`
  - `T`：ウィンドウ長（既定 48）
  - `J`：関節数（既定 17）
  - `C`：チャネル（既定 3：x, y, conf）

詳細は `training/README.md` を参照してください。

## Roadmap

- [ ] 学習データの収集/書き出し（ウィンドウ化キーポイント＋ラベル）
- [ ] 動作フェーズ検出（準備/スイング/フォロースルー）
- [ ] 動作分類の自動化（サーブ / フォア / バック）
- [ ] マルチタスク採点（エラー種別 + 数値スコア）
- [ ] Core ML 統合と端末上評価（遅延/消費電力/精度）

## 免責事項

本プロジェクトはトレーニング支援と MVP 検証のためのものです。専門コーチの指導の代替ではありません。安全に配慮して練習してください。


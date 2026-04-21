//
//  ContentView.swift
//  MotionCoach
//
//  Created by Grassroot on 2026/03/20.
//

import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var cameraManager = CameraSessionManager()

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraPreviewView(
                session: cameraManager.session,
                poseSnapshot: cameraManager.latestPoseOverlay,
                poseOverlayTick: cameraManager.poseOverlayTick
            )
            .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 10) {
                Text(cameraManager.feedback.title)
                    .font(.headline)
                    .foregroundStyle(.white)

                Text(cameraManager.feedback.message)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.95))

                HStack {
                    Label("得分 \(cameraManager.feedback.score)", systemImage: "gauge.with.dots.needle.33percent")
                    Spacer()
                    Label("FPS \(cameraManager.fps)", systemImage: "speedometer")
                }
                .font(.caption)
                .foregroundStyle(.white.opacity(0.9))

                Text(cameraManager.modelStatusText)
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.8))

                Text(cameraManager.actionPredictionText)
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.95))
            }
            .padding(14)
            .background(.black.opacity(0.55))
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .padding(.horizontal, 16)
            .padding(.bottom, 26)
        }
        .overlay(alignment: .top) {
            if !cameraManager.isCameraAuthorized {
                Text("请在系统设置里开启相机权限")
                    .font(.caption)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.red.opacity(0.85))
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
                    .padding(.top, 8)
            }
        }
        .overlay(alignment: .topTrailing) {
            if cameraManager.isCameraAuthorized {
                Button {
                    cameraManager.switchCamera()
                } label: {
                    Image(systemName: "camera.rotate.fill")
                        .font(.title2)
                        .foregroundStyle(.white)
                        .frame(width: 48, height: 48)
                        .background(.black.opacity(0.45), in: Circle())
                }
                .accessibilityLabel(cameraManager.cameraPosition == .front ? "切换到后置摄像头" : "切换到前置摄像头")
                .padding(.trailing, 16)
                .padding(.top, 56)
            }
        }
        .task {
            await cameraManager.prepareAndStart()
        }
        .onDisappear {
            cameraManager.stop()
        }
    }
}

#Preview {
    ContentView()
}

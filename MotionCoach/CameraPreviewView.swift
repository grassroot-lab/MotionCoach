import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let poseSnapshot: PoseOverlaySnapshot?
    /// 每次识别更新递增，用于强制触发 `updateUIView`（避免 SwiftUI/Combine 对“值相等”的跳过）。
    let poseOverlayTick: UInt

    func makeUIView(context: Context) -> PreviewContainerView {
        let container = PreviewContainerView()
        container.previewView.videoPreviewLayer.session = session
        container.previewView.videoPreviewLayer.videoGravity = .resizeAspectFill
        container.skeletonOverlay.previewLayer = container.previewView.videoPreviewLayer
        return container
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        // `poseOverlayTick` 仅用于让 SwiftUI 在值不变时也能触发刷新；这里读取避免 “unused” 警告。
        _ = poseOverlayTick

        if uiView.previewView.videoPreviewLayer.session !== session {
            uiView.previewView.videoPreviewLayer.session = session
        }

        uiView.skeletonOverlay.previewLayer = uiView.previewView.videoPreviewLayer
        uiView.skeletonOverlay.snapshot = poseSnapshot
        uiView.previewView.bringSubviewToFront(uiView.skeletonOverlay)
        uiView.skeletonOverlay.setNeedsDisplay()
    }
}

final class PreviewContainerView: UIView {
    let previewView = PreviewView()
    let skeletonOverlay = PoseSkeletonOverlayView()

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        previewView.translatesAutoresizingMaskIntoConstraints = false
        skeletonOverlay.translatesAutoresizingMaskIntoConstraints = false

        addSubview(previewView)
        previewView.addSubview(skeletonOverlay)

        NSLayoutConstraint.activate([
            previewView.leadingAnchor.constraint(equalTo: leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: trailingAnchor),
            previewView.topAnchor.constraint(equalTo: topAnchor),
            previewView.bottomAnchor.constraint(equalTo: bottomAnchor),

            skeletonOverlay.leadingAnchor.constraint(equalTo: previewView.leadingAnchor),
            skeletonOverlay.trailingAnchor.constraint(equalTo: previewView.trailingAnchor),
            skeletonOverlay.topAnchor.constraint(equalTo: previewView.topAnchor),
            skeletonOverlay.bottomAnchor.constraint(equalTo: previewView.bottomAnchor)
        ])
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        guard let layer = layer as? AVCaptureVideoPreviewLayer else {
            fatalError("Expected AVCaptureVideoPreviewLayer")
        }
        return layer
    }
}

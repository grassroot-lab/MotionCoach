import AVFoundation
import UIKit

/// Draws the human pose skeleton as an overlay on top of `AVCaptureVideoPreviewLayer`.
/// Uses `PoseOverlaySnapshot` to avoid `VNHumanBodyPoseObservation` lifetime issues.
final class PoseSkeletonOverlayView: UIView {
    weak var previewLayer: AVCaptureVideoPreviewLayer?

    var snapshot: PoseOverlaySnapshot? {
        didSet { setNeedsDisplay() }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false
        contentMode = .redraw
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let snapshot, let previewLayer else { return }
        guard let ctx = UIGraphicsGetCurrentContext() else { return }

        ctx.setLineWidth(3)
        ctx.setStrokeColor(UIColor.systemGreen.cgColor)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        for segment in snapshot.segments {
            let la = Self.layerPoint(fromVisionNormalized: segment.a, previewLayer: previewLayer)
            let lb = Self.layerPoint(fromVisionNormalized: segment.b, previewLayer: previewLayer)

            ctx.move(to: la)
            ctx.addLine(to: lb)
            ctx.strokePath()
        }

        ctx.setFillColor(UIColor.systemYellow.cgColor)
        let jointRadius: CGFloat = 5

        for joint in snapshot.joints {
            let lp = Self.layerPoint(fromVisionNormalized: joint, previewLayer: previewLayer)
            let r = CGRect(x: lp.x - jointRadius, y: lp.y - jointRadius, width: jointRadius * 2, height: jointRadius * 2)
            ctx.fillEllipse(in: r)
        }
    }

    /// Vision normalized coordinates use a bottom-left origin, while `fromCaptureDevicePoint`
    /// uses a top-left origin. Convert by flipping Y.
    private static func layerPoint(fromVisionNormalized vision: CGPoint, previewLayer: AVCaptureVideoPreviewLayer) -> CGPoint {
        let devicePoint = CGPoint(x: vision.x, y: 1.0 - vision.y)
        return previewLayer.layerPointConverted(fromCaptureDevicePoint: devicePoint)
    }
}

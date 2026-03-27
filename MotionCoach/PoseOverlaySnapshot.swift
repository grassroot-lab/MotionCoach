import CoreGraphics
import Vision

/// 从 `VNHumanBodyPoseObservation` **同步拷贝**出的绘制数据，避免异步到主线程后 observation 失效导致骨架画不出来。
struct PoseOverlaySnapshot: Equatable {
    struct LineSegment: Equatable {
        let a: CGPoint
        let b: CGPoint
    }

    var segments: [LineSegment]
    var joints: [CGPoint]

    static func build(from observation: VNHumanBodyPoseObservation, confidenceThreshold: Float) -> PoseOverlaySnapshot {
        var segments: [LineSegment] = []
        segments.reserveCapacity(BodyPoseSkeletonSpec.connections.count)

        for (ja, jb) in BodyPoseSkeletonSpec.connections {
            guard
                let pa = try? observation.recognizedPoint(ja),
                let pb = try? observation.recognizedPoint(jb),
                pa.confidence >= confidenceThreshold,
                pb.confidence >= confidenceThreshold
            else { continue }

            segments.append(
                LineSegment(
                    a: CGPoint(x: pa.x, y: pa.y),
                    b: CGPoint(x: pb.x, y: pb.y)
                )
            )
        }

        var joints: [CGPoint] = []
        joints.reserveCapacity(BodyPoseSkeletonSpec.drawableJoints.count)

        for joint in BodyPoseSkeletonSpec.drawableJoints {
            guard
                let p = try? observation.recognizedPoint(joint),
                p.confidence >= confidenceThreshold
            else { continue }
            joints.append(CGPoint(x: p.x, y: p.y))
        }

        return PoseOverlaySnapshot(segments: segments, joints: joints)
    }
}

import Vision

/// Vision 人体姿态：用于绘制骨架的关节连线与关节点集合（与 `VNDetectHumanBodyPoseRequest` 输出一致）。
enum BodyPoseSkeletonSpec {
    static let drawableJoints: [VNHumanBodyPoseObservation.JointName] = [
        .nose, .leftEye, .rightEye, .leftEar, .rightEar,
        .leftShoulder, .rightShoulder, .leftElbow, .rightElbow, .leftWrist, .rightWrist,
        .leftHip, .rightHip, .leftKnee, .rightKnee, .leftAnkle, .rightAnkle,
        .neck, .root
    ]

    static let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        (.nose, .leftEye), (.nose, .rightEye),
        (.leftEye, .leftEar), (.rightEye, .rightEar),
        (.neck, .leftShoulder), (.neck, .rightShoulder),
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip), (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
        (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
        (.root, .leftHip), (.root, .rightHip)
    ]
}

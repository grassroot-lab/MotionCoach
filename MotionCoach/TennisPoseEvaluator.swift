import CoreGraphics
import Vision

struct PoseFeedback {
    let title: String
    let message: String
    let score: Int

    static let waiting = PoseFeedback(
        title: "准备中",
        message: "请把全身放进画面，保持侧身站位。",
        score: 0
    )
}

struct TennisPoseEvaluator {
    func evaluate(observation: VNHumanBodyPoseObservation) -> PoseFeedback {
        guard
            let leftShoulder = point(.leftShoulder, in: observation),
            let rightShoulder = point(.rightShoulder, in: observation),
            let leftHip = point(.leftHip, in: observation),
            let rightHip = point(.rightHip, in: observation),
            let leftElbow = point(.leftElbow, in: observation),
            let rightElbow = point(.rightElbow, in: observation),
            let leftWrist = point(.leftWrist, in: observation),
            let rightWrist = point(.rightWrist, in: observation)
        else {
            return PoseFeedback(
                title: "未识别完整姿态",
                message: "后退半步并确保手臂和躯干都在画面中。",
                score: 20
            )
        }

        let shoulderWidth = distance(leftShoulder, rightShoulder)
        guard shoulderWidth > 0.06 else {
            return PoseFeedback(
                title: "距离不合适",
                message: "请离镜头远一点，便于识别肩部与手臂。",
                score: 25
            )
        }

        let centerX = (leftShoulder.x + rightShoulder.x) * 0.5
        let useRightArm = rightWrist.x > centerX
        let shoulder = useRightArm ? rightShoulder : leftShoulder
        let elbow = useRightArm ? rightElbow : leftElbow
        let wrist = useRightArm ? rightWrist : leftWrist
        let hip = useRightArm ? rightHip : leftHip

        let elbowAngle = angle(a: shoulder, b: elbow, c: wrist)
        let contactDistance = distance(wrist, hip) / shoulderWidth
        let shouldersTilt = abs(leftShoulder.y - rightShoulder.y)

        var score = 100
        var issues: [String] = []

        if elbowAngle < 120 {
            score -= 30
            issues.append("击球时手臂再舒展一些")
        }

        if contactDistance < 0.95 {
            score -= 30
            issues.append("击球点离身体再远一点")
        }

        if shouldersTilt > 0.13 {
            score -= 15
            issues.append("注意保持肩线稳定")
        }

        if wrist.y < shoulder.y - 0.06 {
            score -= 10
            issues.append("随挥时拍头再向上带")
        }

        score = max(0, min(100, score))

        if issues.isEmpty {
            return PoseFeedback(
                title: "动作不错",
                message: "继续保持，下一拍关注重心前移。",
                score: score
            )
        }

        return PoseFeedback(
            title: score >= 70 ? "基本正确" : "需要调整",
            message: issues.prefix(2).joined(separator: "；"),
            score: score
        )
    }

    private func point(_ joint: VNHumanBodyPoseObservation.JointName, in observation: VNHumanBodyPoseObservation) -> CGPoint? {
        guard
            let recognizedPoint = try? observation.recognizedPoint(joint),
            recognizedPoint.confidence > 0.35
        else {
            return nil
        }
        return CGPoint(x: recognizedPoint.x, y: recognizedPoint.y)
    }

    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        hypot(p1.x - p2.x, p1.y - p2.y)
    }

    private func angle(a: CGPoint, b: CGPoint, c: CGPoint) -> CGFloat {
        let ba = CGVector(dx: a.x - b.x, dy: a.y - b.y)
        let bc = CGVector(dx: c.x - b.x, dy: c.y - b.y)
        let denominator = max(0.0001, hypot(ba.dx, ba.dy) * hypot(bc.dx, bc.dy))
        let cosine = max(-1.0, min(1.0, (ba.dx * bc.dx + ba.dy * bc.dy) / denominator))
        return acos(cosine) * 180 / .pi
    }
}

import CoreML
import Foundation
import Vision

struct ActionPrediction {
    let topLabel: String
    let stableLabel: String
    let probabilities: [Float] // fixed order: forehand, one_handed_backhand, two_handed_backhand
    let confidence: Float

    var displayText: String {
        let pct = probabilities.map { Int(round($0 * 100)) }
        return "实时 \(topLabel) (\(Int(round(confidence * 100)))%) · 稳定 \(stableLabel) | FH \(pct[0])% · 1HBH \(pct[1])% · 2HBH \(pct[2])%"
    }
}

final class ForehandActionClassifier {
    private let labels = ["forehand", "one_handed_backhand", "two_handed_backhand"]
    private let windowSize = 48
    private let confidenceThreshold: Float = 0.0
    private let uncertainThreshold: Float = 0.60
    private let smoothingWindow: Int = 8

    private var frameBuffer: [[Float]] = []
    private var labelHistory: [String] = []
    private var model: MLModel?

    init() {
        model = loadModel()
    }

    var statusText: String {
        if model == nil {
            return "模型未加载（请将 MotionCoachActionClassifier.mlpackage 加入 app target）"
        }
        return "模型已加载"
    }

    func predict(observation: VNHumanBodyPoseObservation) -> ActionPrediction? {
        let frame = extractFrame(from: observation)
        frameBuffer.append(frame)
        if frameBuffer.count > windowSize {
            frameBuffer.removeFirst(frameBuffer.count - windowSize)
        }

        guard frameBuffer.count == windowSize, let model else {
            return nil
        }

        let normalized = normalizeWindow(frameBuffer)
        let joints = BodyPoseSkeletonSpec.modelJoints.count
        let channels = 3
        let totalCount = 1 * windowSize * joints * channels

        guard let inputArray = try? MLMultiArray(
            shape: [1, NSNumber(value: windowSize), NSNumber(value: joints), NSNumber(value: channels)],
            dataType: .float32
        ) else {
            return nil
        }

        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(inputArray.dataPointer))
        for i in 0..<totalCount {
            ptr[i] = Float32(normalized[i])
        }

        guard
            let input = try? MLDictionaryFeatureProvider(dictionary: ["keypoints": inputArray]),
            let output = try? model.prediction(from: input),
            let logitsArray = output.featureValue(for: "logits")?.multiArrayValue
        else {
            return nil
        }

        let logits = toFloatArray(logitsArray, expectedCount: labels.count)
        guard logits.count == labels.count else { return nil }
        let probs = softmax(logits)
        guard let top = probs.enumerated().max(by: { $0.element < $1.element }) else {
            return nil
        }
        let topIndex = top.offset
        let topProb = top.element
        let instantLabel = topProb >= uncertainThreshold ? labels[topIndex] : "uncertain"
        let stableLabel = updateStableLabel(with: instantLabel)

        return ActionPrediction(
            topLabel: instantLabel,
            stableLabel: stableLabel,
            probabilities: probs,
            confidence: topProb
        )
    }

    private func loadModel() -> MLModel? {
        // Prefer compiled model if already built by Xcode.
        if let compiledURL = Bundle.main.url(forResource: "MotionCoachActionClassifier", withExtension: "mlmodelc"),
           let m = try? MLModel(contentsOf: compiledURL) {
            return m
        }

        // Fallback: compile .mlpackage at runtime.
        if let packageURL = Bundle.main.url(forResource: "MotionCoachActionClassifier", withExtension: "mlpackage"),
           let compiledURL = try? MLModel.compileModel(at: packageURL),
           let m = try? MLModel(contentsOf: compiledURL) {
            return m
        }
        return nil
    }

    private func extractFrame(from observation: VNHumanBodyPoseObservation) -> [Float] {
        var values = [Float](repeating: 0, count: BodyPoseSkeletonSpec.modelJoints.count * 3)
        var offset = 0
        for joint in BodyPoseSkeletonSpec.modelJoints {
            if let p = try? observation.recognizedPoint(joint), p.confidence >= confidenceThreshold {
                values[offset] = Float(p.x)
                values[offset + 1] = Float(p.y)
                values[offset + 2] = Float(p.confidence)
            }
            offset += 3
        }
        return values
    }

    private func normalizeWindow(_ window: [[Float]]) -> [Float] {
        // Matches training-side normalization:
        // per-frame xy centering + per-frame RMS scaling; keep conf unchanged.
        var out = window
        let jointCount = BodyPoseSkeletonSpec.modelJoints.count
        for t in 0..<out.count {
            var meanX: Float = 0
            var meanY: Float = 0
            for j in 0..<jointCount {
                meanX += out[t][j * 3]
                meanY += out[t][j * 3 + 1]
            }
            meanX /= Float(jointCount)
            meanY /= Float(jointCount)

            var sqSum: Float = 0
            for j in 0..<jointCount {
                let idx = j * 3
                let x0 = out[t][idx] - meanX
                let y0 = out[t][idx + 1] - meanY
                out[t][idx] = x0
                out[t][idx + 1] = y0
                sqSum += (x0 * x0 + y0 * y0)
            }
            let rms = sqrt(max(1e-6, sqSum / Float(jointCount * 2)))
            for j in 0..<jointCount {
                let idx = j * 3
                out[t][idx] /= rms
                out[t][idx + 1] /= rms
                // out[t][idx + 2] keeps confidence unchanged
            }
        }
        return out.flatMap { $0 }
    }

    private func softmax(_ x: [Float]) -> [Float] {
        guard let m = x.max() else { return x }
        let exps = x.map { expf($0 - m) }
        let sum = exps.reduce(0, +)
        if sum <= 0 { return Array(repeating: 0, count: x.count) }
        return exps.map { $0 / sum }
    }

    private func toFloatArray(_ arr: MLMultiArray, expectedCount: Int) -> [Float] {
        switch arr.dataType {
        case .float32:
            let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
            return Array(UnsafeBufferPointer(start: ptr, count: min(expectedCount, arr.count))).map { Float($0) }
        case .double:
            let ptr = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
            return Array(UnsafeBufferPointer(start: ptr, count: min(expectedCount, arr.count))).map { Float($0) }
        default:
            var vals: [Float] = []
            vals.reserveCapacity(min(expectedCount, arr.count))
            for i in 0..<min(expectedCount, arr.count) {
                vals.append(arr[i].floatValue)
            }
            return vals
        }
    }

    private func updateStableLabel(with newLabel: String) -> String {
        labelHistory.append(newLabel)
        if labelHistory.count > smoothingWindow {
            labelHistory.removeFirst(labelHistory.count - smoothingWindow)
        }
        var counts: [String: Int] = [:]
        for label in labelHistory {
            counts[label, default: 0] += 1
        }
        return counts.max(by: { $0.value < $1.value })?.key ?? "uncertain"
    }
}


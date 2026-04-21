import AVFoundation
import Foundation
import Vision

enum ExtractError: Error {
    case usage
    case noVideoTrack
    case cannotStartReader
}

let args = CommandLine.arguments
guard args.count == 3 else {
    fputs("Usage: extract_vision_keypoints.swift <input_video> <output_json>\n", stderr)
    throw ExtractError.usage
}

let inputURL = URL(fileURLWithPath: args[1])
let outputURL = URL(fileURLWithPath: args[2])

let joints: [VNHumanBodyPoseObservation.JointName] = [
    .nose,
    .leftEye, .rightEye,
    .leftEar, .rightEar,
    .leftShoulder, .rightShoulder,
    .leftElbow, .rightElbow,
    .leftWrist, .rightWrist,
    .leftHip, .rightHip,
    .leftKnee, .rightKnee,
    .leftAnkle, .rightAnkle,
]

let asset = AVAsset(url: inputURL)
guard let track = asset.tracks(withMediaType: .video).first else {
    throw ExtractError.noVideoTrack
}

let reader = try AVAssetReader(asset: asset)
let outputSettings: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
]
let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
readerOutput.alwaysCopiesSampleData = false

guard reader.canAdd(readerOutput) else {
    throw ExtractError.cannotStartReader
}
reader.add(readerOutput)

guard reader.startReading() else {
    throw ExtractError.cannotStartReader
}

let request = VNDetectHumanBodyPoseRequest()
var allFrames = [[[Double]]]()

while let sampleBuffer = readerOutput.copyNextSampleBuffer() {
    // This orientation works for most landscape-like videos.
    // If your dataset is portrait and mirrored, adjust to `.right` / `.leftMirrored` as needed.
    let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
    do {
        try handler.perform([request])
    } catch {
        // On error, append a zero frame to keep temporal alignment.
        allFrames.append(Array(repeating: [0.0, 0.0, 0.0], count: joints.count))
        continue
    }

    var frame = Array(repeating: [0.0, 0.0, 0.0], count: joints.count)
    if let obs = request.results?.first {
        for (i, joint) in joints.enumerated() {
            if let p = try? obs.recognizedPoint(joint), p.confidence > 0 {
                frame[i] = [p.x, p.y, Double(p.confidence)]
            }
        }
    }
    allFrames.append(frame)
}

let jsonData = try JSONSerialization.data(withJSONObject: allFrames, options: [])
try jsonData.write(to: outputURL)

print("frames=\(allFrames.count) joints=\(joints.count) out=\(outputURL.path)")

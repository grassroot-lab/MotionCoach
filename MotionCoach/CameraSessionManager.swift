import AVFoundation
import AVFAudio
import Combine
import Foundation
import Vision

final class CameraSessionManager: NSObject, ObservableObject {
    @Published private(set) var feedback: PoseFeedback = .waiting
    @Published private(set) var fps: Int = 0
    @Published private(set) var isCameraAuthorized = true
    @Published private(set) var latestPoseOverlay: PoseOverlaySnapshot?
    /// Increments on each overlay update to force `UIViewRepresentable` refresh.
    @Published private(set) var poseOverlayTick: UInt = 0
    /// Current camera position for UI only; kept in sync with `activeCameraPosition`.
    @Published private(set) var cameraPosition: AVCaptureDevice.Position = .back

    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let outputQueue = DispatchQueue(label: "motioncourt.video.output", qos: .userInitiated)
    private let evaluator = TennisPoseEvaluator()
    private let speechSynthesizer = AVSpeechSynthesizer()

    /// Read/write on the capture queue; used for Vision orientation and camera switching.
    private var activeCameraPosition: AVCaptureDevice.Position = .back

    private var frameCounter = 0
    private var displayedFrames = 0
    private var fpsWindowStart = Date()
    private var lastSpokenAt = Date.distantPast
    private var lastSpokenMessage = ""
    private var requestInFlight = false

    func prepareAndStart() async {
        let granted = await requestCameraPermission()
        isCameraAuthorized = granted
        guard granted else { return }

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            outputQueue.async { [weak self] in
                self?.buildSessionIfNeeded(initialCamera: .back)
                continuation.resume()
            }
        }
        start()
    }

    /// Switch between front/back wide cameras (reconfigures inputs on the capture queue).
    func switchCamera() {
        outputQueue.async { [weak self] in
            guard let self else { return }
            let next: AVCaptureDevice.Position = self.activeCameraPosition == .back ? .front : .back
            self.reconfigureCameraInput(to: next)
        }
    }

    func start() {
        guard !session.isRunning else { return }
        outputQueue.async { [weak session] in
            session?.startRunning()
        }
    }

    func stop() {
        guard session.isRunning else { return }
        outputQueue.async { [weak session] in
            session?.stopRunning()
        }
    }

    private func requestCameraPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func buildSessionIfNeeded(initialCamera: AVCaptureDevice.Position) {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if !session.outputs.contains(where: { $0 === videoOutput }) {
            session.sessionPreset = .high
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
            guard session.canAddOutput(videoOutput) else { return }
            session.addOutput(videoOutput)
        }

        reconfigureCameraInput(to: initialCamera, withinExistingConfiguration: true)
    }

    private func reconfigureCameraInput(to position: AVCaptureDevice.Position, withinExistingConfiguration: Bool = false) {
        let begin = !withinExistingConfiguration
        if begin {
            session.beginConfiguration()
        }
        defer {
            if begin {
                session.commitConfiguration()
            }
        }

        for input in session.inputs {
            session.removeInput(input)
        }

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input)
        else {
            return
        }
        session.addInput(input)
        activeCameraPosition = position

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = (position == .front)
            }
        }

        Task { @MainActor in
            self.cameraPosition = position
        }
    }

    private func maybeSpeak(feedback: PoseFeedback) {
        let now = Date()
        let shouldSpeak = feedback.score < 75 || feedback.title == "动作不错"
        guard shouldSpeak else { return }
        guard now.timeIntervalSince(lastSpokenAt) > 2.8 else { return }
        guard feedback.message != lastSpokenMessage else { return }

        let utterance = AVSpeechUtterance(string: feedback.message)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = 0.48
        speechSynthesizer.speak(utterance)
        lastSpokenAt = now
        lastSpokenMessage = feedback.message
    }

}

extension CameraSessionManager: @preconcurrency AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frameCounter += 1
        displayedFrames += 1

        let now = Date()
        let elapsed = now.timeIntervalSince(fpsWindowStart)
        if elapsed >= 1 {
            let currentFPS = Int(Double(displayedFrames) / elapsed)
            displayedFrames = 0
            fpsWindowStart = now
            Task { @MainActor in
                fps = currentFPS
            }
        }

        if frameCounter % 3 != 0 || requestInFlight {
            return
        }
        requestInFlight = true

        // Keep Vision orientation aligned with preview layer transform.
        // On current iPhone setup, back camera with `.left` avoids 180-degree skeleton inversion.
        let visionOrientation: CGImagePropertyOrientation = activeCameraPosition == .front ? .leftMirrored : .left

        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(
            cmSampleBuffer: sampleBuffer,
            orientation: visionOrientation,
            options: [:]
        )

        do {
            try handler.perform([request])
            let observation = request.results?.first
            requestInFlight = false

            if let observation {
                // Copy keypoints immediately after `perform`: the observation may become invalid across threads.
                let snapshot = PoseOverlaySnapshot.build(from: observation, confidenceThreshold: 0.25)
                let nextFeedback = evaluator.evaluate(observation: observation)

                Task { @MainActor in
                    latestPoseOverlay = snapshot
                    poseOverlayTick += 1
                    feedback = nextFeedback
                    maybeSpeak(feedback: nextFeedback)
                }
            } else {
                Task { @MainActor in
                    latestPoseOverlay = nil
                    poseOverlayTick += 1
                    feedback = PoseFeedback(
                        title: "等待识别",
                        message: "请站在镜头正中，保持完整身体入镜。",
                        score: 15
                    )
                }
            }
        } catch {
            requestInFlight = false
            Task { @MainActor in
                latestPoseOverlay = nil
                poseOverlayTick += 1
                feedback = PoseFeedback(
                    title: "识别异常",
                    message: "请检查光线，或稍后再试。",
                    score: 10
                )
            }
        }
    }
}

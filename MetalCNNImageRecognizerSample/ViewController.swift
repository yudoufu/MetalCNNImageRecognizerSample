//
//  ViewController.swift
//  MetalCNNImageRecognizerSample
//
//  Created by yudoufu on 2017/03/12.
//  Copyright © 2017年 Personal. All rights reserved.
//

import UIKit
import AVFoundation
import MetalKit
import MetalPerformanceShaders

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    let fps: CMTimeScale = 20

    @IBOutlet weak var previewLayerView: UIView!
    @IBOutlet weak var outputLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setupDeviceAndNetwork()
        setupCamera()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.startSession()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        self.stopSession()
    }

    // MARK: AVCaptureMetadataOutputObjectsDelegate
    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?

    private var camera: AVCaptureDevice!

    private func startSession() {
        if !session.isRunning {
            setupPreviewLayer()
            session.startRunning()
        }
    }

    private func stopSession() {
        if session.isRunning {
            session.stopRunning()
        }
    }

    private func setupPreviewLayer() {
        guard previewLayer == nil else {
            print("preview layer already exist")
            return
        }

        guard let layer = AVCaptureVideoPreviewLayer(session: session) else {
            print("layer error")
            return
        }

        // MEMO: view.frame.sizeを取ると、viewWillAppearの場合でも全画面サイズで取れるのでそちらから取る
        layer.frame = CGRect(origin: CGPoint.zero, size: view.frame.size)
        layer.videoGravity = AVLayerVideoGravityResizeAspectFill

        previewLayerView.layer.addSublayer(layer)
        previewLayer = layer
    }

    private func setupCamera() {
        guard let aCamera = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo) else {
            print("no camera")
            return
        }
        camera = aCamera
        session.sessionPreset = AVCaptureSessionPresetInputPriority

        let status = AVCaptureDevice.authorizationStatus(forMediaType: AVMediaTypeVideo)

        switch status {
        case .authorized:
            connectSession(device: camera)
            break
        case .notDetermined:
            AVCaptureDevice.requestAccess(forMediaType: AVMediaTypeVideo, completionHandler: { [weak self] authorized in
                guard let strongSelf = self else { return }
                if  authorized {
                    strongSelf.connectSession(device: strongSelf.camera)
                }
            })
            break
        case .restricted, .denied:
            print("camera not autherized")
            return
        }
    }

    private func connectSession(device: AVCaptureDevice) {
        do {
            try setVideoInput(device: device)
            setVideoOutput()
        } catch let error as NSError {
            print(error)
        }
    }

    private func setVideoInput(device: AVCaptureDevice) throws {
        let videoInput = try AVCaptureDeviceInput(device: device)

        if session.canAddInput(videoInput) {
            session.addInput(videoInput)
        }

        do {
            try camera.lockForConfiguration()
            camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: fps)
            camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: fps)
            camera.unlockForConfiguration()
        } catch {
            fatalError()
        }
    }

    private func setVideoOutput() {
        let videoOutput = AVCaptureVideoDataOutput()
        let queue = DispatchQueue(label: "com.yudoufu.dev.ios")
        //let queue = DispatchQueue.main

        videoOutput.setSampleBufferDelegate(self, queue: queue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [ String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA ]

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        var videoConnection: AVCaptureConnection? = nil

        for connection in videoOutput.connections {
            for port in (connection as! AVCaptureConnection).inputPorts {
                if (port as! AVCaptureInputPort).mediaType == AVMediaTypeVideo {
                    videoConnection = connection as? AVCaptureConnection
                }
            }
        }

        if videoConnection!.isVideoOrientationSupported {
            videoConnection?.videoOrientation = AVCaptureVideoOrientation.portrait
        }

        session.commitConfiguration()
    }

    func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvImageBuffer: imageBuffer)

        fetchImage(ciImage: ciImage)
    }

    //////// Metal CNN /////////////////

    private var network: Inception3Net!

    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var textureLoader: MTKTextureLoader!
    private var sourceTexture: MTLTexture? = nil
    private var ciContext: CIContext!

    private func setupDeviceAndNetwork() {
        guard let aDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal not Supported on current Device")
            return
        }
        device = aDevice

        guard MPSSupportsMTLDevice(device) else {
            print("Metal Performance Shaders not Supported on current Device")
            return
        }
        commandQueue = device.makeCommandQueue()
        textureLoader = MTKTextureLoader(device: device)
        ciContext = CIContext(mtlDevice: device)

        network = Inception3Net(withCommandQueue: commandQueue)
    }

    private func fetchImage(ciImage: CIImage) {
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {return}

        do {
            sourceTexture = try textureLoader.newTexture(with: cgImage, options: [:])
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }
        
        runNetwork()
    }

    private func runNetwork() {
        autoreleasepool() {
            let commandBuffer = commandQueue.makeCommandBuffer()

            network.forward(commandBuffer: commandBuffer, sourceTexture: sourceTexture)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let label = network.getLabel()
            DispatchQueue.main.async {
                self.outputLabel.text = label
            }
        }
    }
}

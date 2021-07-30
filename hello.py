import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
import pyttsx3
engine = pyttsx3.init() # object creation
classes = ["bus front", "bus rear", "bus route", "bus side", "front door", "rear door"]

# creating a pipeline
pipeline = depthai.Pipeline()

# create RGB camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# create a mobilenet ssd
detection_nn = pipeline.createMobileNetDetectionNetwork()
#custom_mobilenet.json
detection_nn.setBlobPath("./nn/custom_mobilenet/frozen_inference_graph.blob")
detection_nn.setConfidenceThreshold(0.5)

# link camera input to mobilenet
cam_rgb.preview.link(detection_nn.input)

# create pipeline output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

engine.startLoop(False)
with depthai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    frame = None
    detections = []

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if cv2.waitKey(1) == ord('q'):
            break

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            
        if in_nn is not None:
            detections = in_nn.detections

        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                print(detection.label, detection.confidence, detection.xmin, detection.xmax, detection.ymin, detection.ymax)
                #TODO find depth and locate distance and orientation
                if not engine.isBusy:
                    engine.say(classes[detection.label])
                    engine.iterate()
                
            cv2.imshow("preview", frame)

engine.endLoop()
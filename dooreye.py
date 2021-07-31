import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai as dai # depthai - access the camera and its data packets
import os
import subprocess
import time

# import pyttsx3
# engine = pyttsx3.init() # object creation
labelMap = ["bus front", "bus rear", "bus route", "bus side", "front door", "rear door"]
#######################
# creating a pipeline
#######################
pipeline = dai.Pipeline()

fullFrameTracking  = False

#------------------
# create RGB camera
#------------------

cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#cam_rgb.setFps(fps)

# -------------------
# create depth camera
# -------------------


## set left input
res = dai.MonoCameraProperties.SensorResolution.THE_400_P
mono_left = pipeline.createMonoCamera()
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_left.setResolution(res)
#mono_left.setFps(fps)

# set right input
mono_right = pipeline.createMonoCamera()
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_right.setResolution(res)
#mono_right.setFps(fps)

# connect left and right to the stereo depth
#https://github.com/luxonis/depthai-experiments/blob/master/gen2-camera-demo/main.py
#https://docs.luxonis.com/projects/api/en/latest/samples/spatial_object_tracker/

stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(200)
stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)

objectTracker = pipeline.createObjectTracker()
objectTracker.setDetectionLabelsToTrack([0,1,2,3,4]) 
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

#-----------------------
# create a mobilenet ssd
#-----------------------

detection_nn = pipeline.createMobileNetSpatialDetectionNetwork()
#custom_mobilenet.json
detection_nn.setBlobPath("./nn/custom_mobilenet/frozen_inference_graph.blob")
detection_nn.setConfidenceThreshold(0.5)
detection_nn.input.setBlocking(False)
detection_nn.setBoundingBoxScaleFactor(0.5)
detection_nn.setDepthLowerThreshold(100)
detection_nn.setDepthUpperThreshold(5000)

# link camera input to mobilenet
cam_rgb.preview.link(detection_nn.input)

#########################
# create pipeline output
#########################

# link the RGB
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Link the depth
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.disparity.link(xout_depth.input)

# link the mobilenet ssd
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

#tracker
trackerOut = pipeline.createXLinkOut()
trackerOut.setStreamName("tracklets")

objectTracker.passthroughTrackerFrame.link(xout_rgb.input)
objectTracker.out.link(trackerOut.input)

if fullFrameTracking:
    cam_rgb.setPreviewKeepAspectRatio(False)
    cam_rgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    detection_nn.passthrough.link(objectTracker.inputTrackerFrame)

detection_nn.passthrough.link(objectTracker.inputDetectionFrame)
detection_nn.out.link(objectTracker.inputDetections)
stereo.depth.link(detection_nn.inputDepth)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    preview = device.getOutputQueue("rgb", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while(True):
        imgFrame = preview.get()
        track = tracklets.tryGet()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = imgFrame.getCvFrame()
        if track is not None:
            trackletsData = track.tracklets
            for t in trackletsData:
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                try:
                    label = labelMap[t.label]
                except:
                    label = t.label

                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break

# with dai.Device(pipeline) as device:
#     q_rgb = device.getOutputQueue("rgb")
#     q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
#     q_nn = device.getOutputQueue("nn")

#     frame = None
#     dframe = None
#     detections = []

#     while True:
#         in_rgb = q_rgb.tryGet()
#         in_depth = q_depth.tryGet()
#         in_nn = q_nn.tryGet()

#         if cv2.waitKey(1) == ord('q'):
#             break

#         if in_rgb is not None:
#             frame = in_rgb.getCvFrame()

#         if in_depth is not None:
#             dframe = in_depth.getFrame()
#             # Normalization for better visualization
#             dframe = (dframe * (255 / stereo.getMaxDisparity())).astype(np.uint8)

#             # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
#             dframe = cv2.applyColorMap(dframe, cv2.COLORMAP_JET)

            
#         if in_nn is not None:
#             detections = in_nn.detections

#         if frame is not None:
#             for detection in detections:
#                 bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#                 #print(bbox)
#                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
#                 cv2.putText(frame, classes[detection.label], (bbox[0]+3, bbox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
#                 #print(detection.label, detection.confidence, detection.xmin, detection.xmax, detection.ymin, detection.ymax)
#                 #TODO find depth and locate distance and orientation
#                 # out = subprocess.Popen(['pacmd', 'list-sink-inputs', '|' , 'grep', '-c', "'state: RUNNING'"], 
#                 #                         stdout=subprocess.PIPE, 
#                 #                         stderr=subprocess.STDOUT)
#out = subprocess.Popen(['pactl', 'list', '|', 'grep', '"State: RUNNING"'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#                 # stdout, stderr = out.communicate()
#                 # if stdout==0:
#                 #     cmd = f'pico2wave -w speech.wav "{detection.label}" | aplay'
#                 #     os.system(cmd)
#                 # else:
#                 #     print("audio is playing currently")

#                 if dframe is not None:
#                     d = dframe[bbox[0]:bbox[2], bbox[1]:bbox[3]].flatten()
#                     d = np.around(d).astype(int)
#                     #print(d)
#                     dist = round(np.bincount(d).argmax()/10)
#                     cv2.putText(frame, f"{dist} cm", (bbox[0]+3, bbox[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                    

                
#             cv2.imshow("RGB", frame)
            
#         if dframe is not None:
#             cv2.imshow("disparity_color", dframe)

#!/usr/bin/env python3
# https://raw.githubusercontent.com/luxonis/depthai-python/main/examples/spatial_object_tracker.py
# Copied from the above link and modified by Abhinav Dayal
from logger import logger
import cv2
import depthai as dai
import numpy as np
import time
import csv
from deduplicator import findunique
#from monoculardepth.inference import MonocularDepth

labelMap = ["NONE", "bus", "front door", "rear door", "route"]
#monocularDepth = MonocularDepth()

def setupPipeline(nnPath, fullFrameTracking, input_width, input_height, FPS):
    # Create pipeline
    pipeline = dai.Pipeline()
    confidence_threshold = 0.3
    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    objectTracker = pipeline.createObjectTracker()

    xoutRgb = pipeline.createXLinkOut()
    trackerOut = pipeline.createXLinkOut()

    xoutRgb.setStreamName("preview")
    trackerOut.setStreamName("tracklets")


    camRgb.setPreviewSize(input_width, input_height)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(FPS)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoLeft.setFps(FPS)
    monoRight.setFps(FPS)

    # setting node configs
    stereo.initialConfig.setConfidenceThreshold(255)

    spatialDetectionNetwork.setBlobPath(nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(confidence_threshold)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    objectTracker.setDetectionLabelsToTrack([1,2,4])
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(trackerOut.input)

    if fullFrameTracking: # better keep it false
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.video.link(objectTracker.inputTrackerFrame)
        objectTracker.inputTrackerFrame.setBlocking(False)
        # do not block the pipeline if it's too slow on full frame
        objectTracker.inputTrackerFrame.setQueueSize(2)
    else:
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    return pipeline

H = 10
F = 350
def run(pipeline, input_width, input_height, FPS, outfilecnt=0, bd=None):
    #global monocularDepth
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        preview = device.getOutputQueue("preview", 4, False)
        tracklets = device.getOutputQueue("tracklets", 20, False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'run/output{outfilecnt}.avi', fourcc, FPS, (input_width, input_height))
        out.set(cv2.CAP_PROP_FPS, FPS)
        csvout = open(f'run/output{outfilecnt}.csv', 'w')
        csvwriter = csv.writer(csvout)
        csvwriter.writerow(["frame", "label", "x1", "y1", "x2", "y2", "x", "y", "z", "d"])
        fcounter = 0
        # if bd is not None:
        #     logger.info(f"Bluedot pressed = {bd.is_pressed}, Bluedot connected = {bd.is_connected}, Bluedot running = {bd.running}")
        while bd.is_connected if bd else True:
            imgFrame = preview.get()
            track = tracklets.get()

            counter+=1
            fcounter += 1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            frame = imgFrame.getCvFrame()
            #monodepth = monocularDepth.run_inference(frame)
            #displaydepth = cv2.applyColorMap(monodepth, cv2.COLORMAP_MAGMA)
            trackletsData = track.tracklets
            detections = findunique([{
                "label": t.label, 
                "status": t.status.name, 
                "roi": t.roi.denormalize(frame.shape[1], frame.shape[0]), 
                "x":int(t.spatialCoordinates.x), 
                "y":int(t.spatialCoordinates.y), 
                "z":int(t.spatialCoordinates.z), 
                "id":t.id} 
                for t in trackletsData if t.status.name in ['NEW', 'TRACKED']])

            
            #print(detections)
            if detections:
                for d in detections:
                    roi = d["roi"]#.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    


                    try:
                        label = labelMap[d["label"]]
                    except:
                        label = d["label"]

                    lcolor = color #(0,255,0)
                    lscale = 0.4

                    cv2.putText(frame, str(label).upper(), (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    cv2.putText(frame, f"ID: {d['id']}", (x1 + 10, y1 + 60), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    cv2.putText(frame, d["status"], (x1 + 10, y1 + 70), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    
                    #dval = np.bincount(monodepth[y1:y2, x1:x2].flatten()).argmax()
                    #cv2.putText(displaydepth, str(label).upper(), (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    #cv2.putText(displaydepth, f"ID: {d['id']}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    #cv2.putText(displaydepth, f"DVAL: {dval}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    #cv2.rectangle(displaydepth, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    cv2.putText(frame, f"X: {d['x']/10} cm", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    cv2.putText(frame, f"Y: {d['y']/10} cm", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    cv2.putText(frame, f"Z: {d['z']/10} cm", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                    if d["label"]==1:
                        p = y2-y1
                        #print(p)
                        caliberated_depth = (H*F)/p
                        cv2.putText(frame, f"Distance: {caliberated_depth:0.2f} m", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
                        csvwriter.writerow([fcounter, str(label).upper(), x1, y1, x2, y2, d['x'], d["y"], d["z"], round(caliberated_depth, 2)])
                    else:
                        csvwriter.writerow([fcounter, str(label).upper(), x1, y1, x2, y2, d['x'], d["y"], d["z"], 0])

                    #cmd = f'pico2wave -w speech.wav "{str(label)} is located 10 centimeter to your left and 100 centimeter in front" | aplay'
                    #os.system(cmd)

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            #display = cv2.hconcat([frame, displaydepth])
            #cv2.imshow("tracker", display)
            cv2.imshow("tracker", frame)
            
            # output the frame
            #out.write(display)
            out.write(frame)
            #logger.info("writing frame")

            if cv2.waitKey(1) == ord('q'):
                break

        # After we release our webcam, we also release the output
        out.release() 
        csvout.close()

if __name__ == "__main__":
    from pathlib import Path
    nnPath = str((Path(__file__).parent / Path('nn/custom_mobilenet/frozen_inference_graph.blob')).resolve().absolute())
    fps = 10
    input_width = 300
    input_height = 300
    fullFrameTracking = False

    pipeline = setupPipeline(nnPath, fullFrameTracking, input_width, input_height, fps)
    run(pipeline, input_width, input_height, fps)
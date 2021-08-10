#!/usr/bin/env python3
# https://raw.githubusercontent.com/luxonis/depthai-python/main/examples/spatial_object_tracker.py
# Copied from the above link and modified by Abhinav Dayal
from logger import logger
import cv2
import depthai as dai
import numpy as np
import time
import csv
from deduplicator import findunique, FilterObjectsByBus, areatheshold

#from monoculardepth.inference import MonocularDepth

labelMap = ["NONE", "auto", "bus", "bus stop", "car", "driver door", "front door", "person", "rear door", "route", "truck", "two wheeler"]
#monocularDepth = MonocularDepth()

def setupPipeline(nnPath, fullFrameTracking, input_width, input_height, FPS):
    # Create pipeline
    pipeline = dai.Pipeline()
    confidence_threshold = 0.1
    bus_confidence_threshold = 0.4

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

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    spatialDetectionNetwork.out.link(xout_nn.input)


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
    spatialDetectionNetwork.setDepthUpperThreshold(50000)

    objectTracker.setDetectionLabelsToTrack([2]) #[1,2,3,4,5,6,7,8,9,10,11]
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)
    # Maximum number of object to track. Maximum 60.
    objectTracker.setMaxObjectsToTrack(20)

    # Above this threshold the detected objects will be tracked. Default 0, all image detections are tracked.
    objectTracker.setTrackerThreshold(bus_confidence_threshold)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(trackerOut.input)

    if fullFrameTracking: # better keep it false
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.video.link(objectTracker.inputTrackerFrame)
        #objectTracker.inputTrackerFrame.setBlocking(False)
        # do not block the pipeline if it's too slow on full frame
        #objectTracker.inputTrackerFrame.setQueueSize(2)
    else:
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

def displayRect(frame, csvwriter, d, fcounter, key):
    x1 = d["minx"]
    y1 = d["miny"]
    x2 = d["maxx"]
    y2 = d["maxy"]

    try:
        label = labelMap[d["label"]]
    except:
        label = d["label"]

    lcolor = { # BGR format
        1: (95, 181, 255), 
        2: (58,72,255), 
        3: (99, 31, 158), 
        4: (133, 167, 7), 
        5: (159, 187, 10),
        6:(47, 214, 255), 
        7:(29, 30, 200), 
        8:(205, 161, 23), 
        9: (40, 90, 240), 
        11: (58, 72, 255), 
        10:(20, 96, 122)}
    lscale = 0.3
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), lcolor[d["label"]], cv2.FONT_HERSHEY_SIMPLEX)
    (w, h), _ = cv2.getTextSize(str(label).title(), cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
    cv2.rectangle(frame, (x1, y1-20), (x1+w, y1), lcolor[d["label"]], -1)
    cv2.putText(frame, str(label).title(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, lscale, (255, 255, 255))
    #cv2.putText(frame, f"Confidence: {d['confidence']}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
    #cv2.putText(frame, f"X: {d['x']/10} cm", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
    #cv2.putText(frame, f"Y: {d['y']/10} cm", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
    #cv2.putText(frame, f"Z: {d['z']/10} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
    
    if d["label"]==2: # if bus
        #cv2.putText(frame, f"ID: {d['id']}", (x1 + 10, y1 + 60), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
        #cv2.putText(frame, f"Distance: {d['depth']:0.2f} m", (x1 + 10, y1 + 70), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
        #cv2.putText(frame, d["status"], (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
        csvwriter.writerow([fcounter, d['id'], str(label).upper(), d["confidence"], x1, y1, x2, y2, d['x'], d["y"], d["z"], round(d['depth'])])
    elif key=='bus': # if not a bus
        #cv2.putText(frame, f"BUSID: {d['busid']}", (x1 + 10, y1 + 60), cv2.FONT_HERSHEY_TRIPLEX, lscale, lcolor)
        csvwriter.writerow([fcounter, d['busid'], str(label).upper(), d["confidence"], x1, y1, x2, y2, d['x'], d["y"], d["z"], 0])
    else:
        csvwriter.writerow([fcounter, 0, str(label).upper(), d["confidence"], x1, y1, x2, y2, d['x'], d["y"], d["z"], round(d['depth']), 0])

    #cmd = f'pico2wave -w speech.wav "{str(label)} is located 10 centimeter to your left and 100 centimeter in front" | aplay'
    #os.system(cmd)

def calc_depth(d):
    H = {1:1.4, 2:2.7, 4:1.5, 7:1.5, 10:3, 11: 1}
    F = 350
    p = d['maxy'] - d['miny']
    caliberated_depth = H.get(d['label'], 0)*F/p
    d["depth"] = caliberated_depth

def run(pipeline, input_width, input_height, FPS, alerts, outfilecnt=0, bd=None):
    #bus_confidence_threshold = 0.6
    
    #global monocularDepth
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        preview = device.getOutputQueue("preview", 4, False)
        tracklets = device.getOutputQueue("tracklets", 20, False)
        q_nn = device.getOutputQueue("nn", 30, False)
        q_depth = device.getOutputQueue("depth", 4, False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'run/output{outfilecnt}.mp4', fourcc, FPS, (input_width, input_height))
        out.set(cv2.CAP_PROP_FPS, FPS)
        csvout = open(f'run/output{outfilecnt}.csv', 'w')
        csvwriter = csv.writer(csvout)
        csvwriter.writerow(["frame", "id", "label", "confidence", "x1", "y1", "x2", "y2", "x", "y", "z", "d"])
        fcounter = 0
        # if bd is not None:
        #     logger.info(f"Bluedot pressed = {bd.is_pressed}, Bluedot connected = {bd.is_connected}, Bluedot running = {bd.running}")
        while bd.is_connected if bd else True:
            wk = cv2.waitKey(1)

            if wk == ord('q'):
                break
            elif wk==ord('p'):
                alerts.prevMode()
            elif wk==ord('n'):
                alerts.nextMode()
            elif wk==ord(' '):
                alerts.sayMessage()

            if alerts.getAlertMode() == "Sleep":
                continue
            imgFrame = preview.tryGet()

            if imgFrame is None:
                continue
            
            track = tracklets.tryGet()
            in_nn = q_nn.tryGet()
            in_depth = q_depth.tryGet()

           

            counter+=1
            fcounter += 1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
            frame = imgFrame.getCvFrame()
            #turns.process(frame)
            #monodepth = monocularDepth.run_inference(frame)
            #displaydepth = cv2.applyColorMap(monodepth, cv2.COLORMAP_MAGMA)
            if track is not None:
                trackletsData = track.tracklets
                # for t in trackletsData:
                #     print(t.TrackingStatus, t.id, t.label, t.roi, t.spatialCoordinates, t.srcImgDetection.confidence, t.status)

                tracked = []
                for t in trackletsData:
                    #t.srcImgDetection.confidence>=bus_confidence_threshold and 
                    if t.status.name in ['NEW', 'TRACKED']:
                        roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                        tracked.append(
                            {
                                "label": t.label, 
                                "status": t.status.name, 
                                "minx": int(roi.topLeft().x), 
                                "maxx": int(roi.bottomRight().x), 
                                "maxy": int(roi.bottomRight().y), 
                                "miny": int(roi.topLeft().y), 
                                "x":int(t.spatialCoordinates.x), 
                                "y":int(t.spatialCoordinates.y), 
                                "z":int(t.spatialCoordinates.z), 
                                "confidence": round(t.srcImgDetection.confidence, 2),
                                "id":t.id
                            }
                        )

                detections = findunique([t for t in tracked if areatheshold(t)])
            else:
                detections = []
            if in_nn is not None:
                objects = [{
                    "label": t.label,
                    "minx": int(t.xmin*frame.shape[1]), 
                    "maxx": int(t.xmax*frame.shape[1]), 
                    "miny": int(t.ymin*frame.shape[0]), 
                    "maxy": int(t.ymax*frame.shape[0]), 
                    "x":int(t.spatialCoordinates.x), 
                    "y":int(t.spatialCoordinates.y), 
                    "z":int(t.spatialCoordinates.z), 
                    "confidence": round(t.confidence, 2)
                } for t in in_nn.detections if t.label in [1,3,4,6,7,8,9,11]]

                objects = [t for t in objects if areatheshold(t)]
                objects = findunique(objects)
            else:
                objects = []
            busobjects = FilterObjectsByBus(detections, objects) # bust dict containing its assets
            objects = {
                "bus": busobjects, 
                "person": [o for o in objects if o["label"]==7 and o["confidence"]>0.15],
                "vehicle": [o for o in objects if o["label"] in [1,4,11] and o["confidence"]>0.15],
                "busstop": [o for o in objects if o["label"]==3 and o["confidence"]>0.15]
            }

            #TODO: comment display by a flag
            app_mode = alerts.getAlertMode()
            """
            0: "Sleep",
            1: "Bus Stop",
            2: "Person",
            3: "Road",
            4: "Bus",
            5: "Bus Door"
            """
            if detections:
                for d in detections:
                    calc_depth(d)
                    if (app_mode == "Bus" or app_mode=="Bus Door"):
                        displayRect(frame, csvwriter, d, fcounter, "bus")
            for key in objects:
                if key == "bus":
                    if app_mode == "Bus" or app_mode=="Bus Door":
                        for d in [items[1] for v in objects[key].items() if v[0]!='bus' for items in v[1].items()]: # flattened list of bus objects
                            displayRect(frame, csvwriter, d, fcounter, key)
                else:
                    for d in objects[key]:
                        calc_depth(d)
                        if (app_mode == "Person" and d["label"]==7) \
                            or (app_mode == "Road" and d["label"] in [1,4,11])\
                            or (app_mode == "Bus Stop" and d["label"] == 3):
                            displayRect(frame, csvwriter, d, fcounter, key) # TODO customize 
                    

            alerts.process(objects, imgFrame.getCvFrame(), in_depth.getFrame() if in_depth is not None else None)
            cv2.putText(frame, "F: {:d}, fps: {:.2f}".format(fcounter, fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
            #display = cv2.hconcat([frame, displaydepth])
            #cv2.imshow("tracker", display)
            cv2.imshow("tracker", frame)
            
            # output the frame
            #out.write(display)
            out.write(frame)
            #logger.info("writing frame")
            

        # After we release our webcam, we also release the output
        out.release() 
        csvout.close()

if __name__ == "__main__":
    from pathlib import Path
    from alertservice import AlertService
    #from turnservice import TurnService
    nnPath = str((Path(__file__).parent / Path('nn/custom_mobilenet/frozen_inference_graph.blob')).resolve().absolute())
    fps = 10
    input_width = 300
    input_height = 300
    fullFrameTracking = False

    pipeline = setupPipeline(nnPath, fullFrameTracking, input_width, input_height, fps)
    alerts = AlertService()
    #turns = TurnService()
    run(pipeline, input_width, input_height, fps, alerts)#, turns)
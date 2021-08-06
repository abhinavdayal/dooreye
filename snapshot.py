from bluedot import BlueDot
from signal import pause
from logger import logger

import cv2
import depthai as dai

pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("preview")
camRgb.preview.link(xoutRgb.input)

input_width = 300
input_height = 300

camRgb.setPreviewSize(input_width, input_height)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

device = dai.Device(pipeline)
preview = device.getOutputQueue("preview", 4, False)
filenum = 0
def snapshot():
    global filenum
    imgFrame = preview.get()
    frame = imgFrame.getCvFrame()
    filenum += 1
    cv2.imwrite(f"./snaps/{filenum}.png", frame)


bd = BlueDot(print_messages=True)

def onBluetoothConnect():
    logger.info("Door Eye Armed")

def onBluetoothDisconnect():
    logger.info("Door Eye Disarmed")

bd.set_when_client_connects(onBluetoothConnect)
bd.set_when_client_disconnects(onBluetoothDisconnect)

# def start_dai():
#     global bd
#     logger.info("starting to capture")
#     tracker.run(pipeline, input_width, input_height, fps, bd)

# def stop_dai():
#     logger.info("goodbye")

bd.when_pressed = snapshot
# bd.when_released = stop_dai

pause()

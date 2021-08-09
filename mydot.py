from bluedot import BlueDot
from signal import pause
from logger import logger
import argparse
from pathlib import Path
import os
import tracker
from alertservice import AlertService

nnPathDefault = str((Path(__file__).parent / Path('nn/custom_mobilenet/frozen_inference_graph.blob')).resolve().absolute())

parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)
parser.add_argument('-iw', '--input_width', help="input width", default=300)
parser.add_argument('-ih', '--input_height', help="input height", default=300)
parser.add_argument('-f', '--fps', help="frame rate", default=10)

args = parser.parse_args()

fps = args.fps
# Properties
input_width = args.input_width
input_height = args.input_height
fullFrameTracking = args.full_frame
nnPath = args.nnPath
alertService = AlertService(fps = fps, duration = 60, gap = 1)

pipeline = tracker.setupPipeline(nnPath, fullFrameTracking, input_width, input_height, fps)
# clean the run
os.system("rm run/*")

bd = BlueDot(print_messages=True, cols=1, rows=3)
outfilecnt = 0
def onBluetoothConnect():
    global bd
    global outfilecnt
    global alertService
    logger.info("Door Eye Armed")
    outfilecnt += 1
    tracker.run(pipeline, input_width, input_height, fps, alertService, outfilecnt, bd)

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

# bd.when_pressed = start_dai
# bd.when_released = stop_dai


def on_prev():
    global alertService
    alertService.nextMode()

def on_next():
    global alertService
    alertService.prevMode()

def on_speak():
    global alertService
    alertService.sayMessage()


bd[0,0].when_pressed = on_prev
bd[0,1].when_pressed = on_next
bd[0,2].when_pressed = on_speak

pause()
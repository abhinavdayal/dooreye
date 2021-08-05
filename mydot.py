from bluedot import BlueDot
from signal import pause
from logger import logger
import argparse
from pathlib import Path
import os
import tracker

nnPathDefault = str((Path(__file__).parent / Path('nn/custom_mobilenet/frozen_inference_graph.blob')).resolve().absolute())

parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

args = parser.parse_args()

fullFrameTracking = args.full_frame
nnPath = args.nnPath
pipeline = tracker.setupPipeline(nnPath, fullFrameTracking)
# clean the run
os.system("rm run/*")

def onBluetoothConnect():
    logger.info("Door Eye Armed")

def onBluetoothDisconnect():
    logger.info("Door Eye Disarmed")

bd = BlueDot(print_messages=True)

bd.set_when_client_connects(onBluetoothConnect)
bd.set_when_client_disconnects(onBluetoothDisconnect)

def start_dai():
    tracker.run(pipeline, bd)

def stop_dai():
    logger.info("goodbye")

bd.when_pressed = start_dai
bd.when_released = stop_dai

pause()
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
parser.add_argument('-iw', '--input_width', help="input width", default=300)
parser.add_argument('-ih', '--input_height', help="input height", default=300)
parser.add_argument('-f', '--fps', help="frame rate", default=30)

args = parser.parse_args()

fps = args.fps
# Properties
input_width = args.input_width
input_height = args.input_height
fullFrameTracking = args.full_frame
nnPath = args.nnPath

pipeline = tracker.setupPipeline(nnPath, fullFrameTracking, input_width, input_height, fps)
# clean the run
os.system("rm run/*")

class KB:
    def __init__(self):
        pass

    @property
    def is_pressed(self):
        return True

kb = KB()
tracker.run(pipeline, input_width, input_height, fps, kb)
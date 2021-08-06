# Copyright 2020 Filippo Aleotti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import cv2
import numpy as np
from . import network
from tensorflow.python.util import deprecation
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class MonocularDepth:
    _instance = None
    network_params = {"height": 320, "width": 640, "is_training": False}
    model = network.Pydnet(network_params)
    tensor_image = tf.placeholder(tf.float32, shape=(320, 640, 3))
    batch_img = tf.expand_dims(tensor_image, 0)
    tensor_depth = model.forward(batch_img)
    tensor_depth = tf.nn.relu(tensor_depth)

    # restore graph
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "monoculardepth/ckpt/pydnet")

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def __init__(self):
        # disable future warnings and info messages for this demo
        pass

    def run_inference(self, frame):
        # preparing image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img = cv2.resize(img, (640, 320))
        img = img / 255.0
        # inference
        depth = self.sess.run(self.tensor_depth, feed_dict={self.tensor_image: img})
        depth = np.squeeze(depth)
        min_depth = depth.min()
        max_depth = depth.max()
        depth = (depth - min_depth) / (max_depth - min_depth)
        depth = (depth*255.0).astype(np.uint8)
        #depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        # preparing final depth
        depth = cv2.resize(depth, (w, h))
        return depth
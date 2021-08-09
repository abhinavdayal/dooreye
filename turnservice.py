#https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
from queue import Queue
import cv2 
import math

class TurnService:
    def __init__(self, duration=60, fps=10, F=350):
        """
        TODO: we may need to be a bit dynamic about orientation
        duration = how many seonds worth history to maintain
        fps = frames per second
        F = The caliberated observed focal length
        """
        self.duration = duration
        self.fps = fps
        self.record = Queue(duration*fps)
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        self.hangle = 0
        self.vangle = 0
        self.F = 350

    def reset(self):
        self.hangle = 0
        self.vangle = 0

    def angle(self, v1, v2):
        s = -1 if v2<v1 else 1
        return s*math.asin(abs(v2 - v1)/self.F)*180/math.pi

    def process(self, img):
        prev = self.record.top()
        self.record.enqueue(img)

        if prev is not None:
            try:
                keypoints_1, descriptors_1 = self.sift.detectAndCompute(prev,None)
                keypoints_2, descriptors_2 = self.sift.detectAndCompute(img,None)
                matches = self.bf.match(descriptors_1,descriptors_2)
                smallest_match = min(matches, key = lambda x:x.distance)
                #img3 = cv2.drawMatches(prev, keypoints_1, img, keypoints_2, matches[:50], img, flags=2)
                #cv2.imshow("sift", img3)
                #for match in matches:
                p1 = keypoints_1[smallest_match.queryIdx].pt
                p2 = keypoints_2[smallest_match.trainIdx].pt
                hangle = self.angle(p1[0], p2[0])
                vangle = self.angle(p1[1], p2[1])
                self.hangle += hangle
                self.vangle += vangle
                print("D:",smallest_match.distance, "Horizontal:", int(p2[0] - p1[0]), "Vertical:", int(p2[1] - p1[1]), "hangle:", self.hangle, "vangle", self.vangle)
            except:
                pass
            #break


    
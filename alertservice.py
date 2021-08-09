from queue import Queue
import numpy as np
import cv2
import math
import os
from logger import logger

def getDoorMesage(coords):
    horiz = "left" if coords.x<100 else "right" if coords.x>100 else None
    if horiz:
        return f"Bus door is located {round(coords.z/40)} steps to your front and about {round(coords.x/40)} steps to your {horiz}."
    else:
        return f"Bus door is located {round(coords.z/40)} steps to your front."

def getBusMessage(coords):
    pass


class AlertService:
    ALERT_MODES = {
        0: "Bus Stop",
        1: "Bus",
        2: "Bus Door",
        3: "Road",
        4: "Person",
        5: "Sleep"
    }

    def __init__(self, fps = 10, duration = 60, gap = 1):
        """
        fps = frames per second
        duration = how many seconds history to keep
        gap = minimum, how many seconds to look behind for object tracking
        """
        self.iteration = 0
        self.fps = fps
        self.duration = duration
        self.gap = gap
        self.record = Queue(duration*fps)
        self.depth = None
        self.frame = None
        self.pframe = None
        self.resetOrientation()
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        self.alertmode = 1 # BUS
        self.tellMode()
        self.message = ""
        self.messagedelivered = False

    def sayMessage(self, must=False):
        if must or not self.messagedelivered:
            self.speak(self.message)
            self.messagedelivered = True

    def updateMessage(self, message):
        self.message = message

    def resetOrientation(self):
        self.hangle = 0
        self.vangle = 0

    def angle(self, v1, v2):
        s = -1 if v2<v1 else 1
        return s*math.asin(abs(v2 - v1)/self.F)*180/math.pi

    def updateOrientation(self):
        prev = self.pframe
        img = self.frame

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
                #print("D:",smallest_match.distance, "Horizontal:", int(p2[0] - p1[0]), "Vertical:", int(p2[1] - p1[1]), "hangle:", self.hangle, "vangle", self.vangle)
            except:
                pass
        
    def __nearestEntity(self, entity):
        objects = self.record.top()
        nearest = None
        for v in objects[entity]:
            v["area"] = (v['maxy']-v['miny'])*(v['maxx']- v['minx'])
            if not nearest or nearest["area"]<v["area"]:
                nearest = v
        return nearest

    def __nearestbus(self):
        objects = self.record.top()
        nearestbus = None
        for _, bus in objects["bus"].items():
            bus["bus"]["area"] = (bus["bus"]['maxy']-bus["bus"]['miny'])*(bus["bus"]['maxx']- bus["bus"]['minx'])
            if not nearestbus or nearestbus["bus"]["area"]<bus["bus"]["area"]:
                nearestbus = bus
        return nearestbus

    def process(self, objects, frame, depth):
        self.record.enqueue(objects)
        self.depth = depth
        self.pframe = self.frame
        self.frame = frame
        self.updateOrientation()
        if self.alertmode == 0:
            self.busStopStatus()
        elif self.alertmode == 1:
            self.busStatus()
        elif self.alertmode == 2:
            self.busDoorStaus()
        elif self.alertmode == 3:
            self.roadStatus()
        else:
            self.personStatus()

        self.iteration += 1

    def getAlertMode(self):
        return self.ALERT_MODES[self.alertmode]

    def nextMode(self):
        self.alertmode = (self.alertmode+1)%len(self.ALERT_MODES)
        self.tellMode()
        self.resetOrientation()

    def prevMode(self):
        self.alertmode = (len(self.ALERT_MODES) + self.alertmode-1)%len(self.ALERT_MODES)
        self.tellMode()
        self.resetOrientation()

    def tellMode(self):
        mode = self.getAlertMode()
        if mode=='Sleep':
            self.speak("Now going to sleep")
        else:
            self.speak(f"Now detecting {mode}")

    def speak(self, message):
        cmd = f'pico2wave -w speech.wav "{message}" | aplay'
        logger.info(message)
        os.system(cmd) 

    def busStatus(self):
        history = self.gap*self.fps
        if(self.record.size()>history): # gap duration passed
            # find the nearest bus in view
            nearestbus = self.__nearestbus()
            # if nearestbus is not None:
            #     print("NEAREST", nearestbus)
            if nearestbus:
                # find from history this bus id
                prev = self.record.peek(-history, nearestbus["bus"]["id"])
                
                if prev is not None:
                    #print("PREV", prev)
                    # check for bus
                    # cid = set(objects["bus"].keys())
                    # pid = set(prev["bus"].keys())
                    # commonids = cid.intersection(pid)
                    #for id in commonids:
                    c = nearestbus["bus"]
                    p = prev["bus"]
                    #print(id, p["depth"], c["depth"])
                    if p["depth"] - c["depth"] > 0.8:
                        self.message =  f"A bus is approaching at a distance of {c['depth']*2} steps."
                    elif p["depth"] - c["depth"] < -0.8:
                        self.message = f"A bus is leaving"
                    else:
                        self.message = f"A Bus is standing in front of you at less than {c['depth']*2} steps."
                    self.resetOrientation()

            else:
                self.message = "No bus found yet. Ensure you are facing the road with incoming direction"

    def busDoorStaus(self, bus):
        """
        From the nearest bus locate the door and tell about the direction
        """
        nearestbus = self.__nearestbus()
        door = nearestbus.get(6, False) or nearestbus.get(8, False)
        if not door:
            self.message = "No door found from current view. Slowly turn a little left and right"
        else:
            self.resetOrientation()
            x = door['x']
            self.message = f"Door is in your front."
            if x<100:
                self.message = f"{self.message} Slightly turn left"
            elif x>100:
                self.message = f"{self.message} Slightly turn right"

    #TODO: GPS based later
    def busStopStatus(self):
        """
        check for bus stops. May be we need to store the last bus stop found as well.
        """
        nearestbusstop = self.__nearestEntity("busstop")
        if nearestbusstop is not None:
            self.message = "There is a busstop in front of you"
            self.resetOrientation()
        else:
            self.message = "No busstop found. Turn around slowly to locate."

    #TODO: compass and GPS can be used here later.
    def roadStatus(self):
        """
        check for road status, may be we need to remember the road and direct turn accordingly
        """
        nearestvehicle = self.__nearestEntity("vehicle")
        roadfound = False
        if nearestvehicle is not None:
            roadfound = True
            
        else:
            depth_a = self.depth.flatten()
            mean, mode = np.average(depth_a), np.argmax(np.bincount(depth_a))
            if mean > 4000 and mode>5000:
                roadfound = True
        if roadfound:
            self.message("You are facing the road. Turn right till you dont see road and then turn left.")
            self.resetOrientation()
        else:
            self.message("You are not facing the road. Turn around slowly to detect.")

    def personStatus(self):
        """
        find nearest person
        """
        nearestperson = self.__nearestEntity("person")

        if nearestperson is not None:
            # person found
            z = int(nearestperson["z"]/1000)
            d = int(nearestperson["depth"])
            self.message(f"There is a person about {d*2} steps in front.")
            x = nearestperson["x"]
            if x<100:
                self.message = f"{self.message}. Turn little to left"
            elif x>100:
                self.message = f"{self.message}. Turn little to right"
            self.resetOrientation()


        

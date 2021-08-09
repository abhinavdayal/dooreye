from queue import Queue
import numpy as np
import cv2
import math
import os
from logger import logger
import time

class AlertService:
    ALERT_MODES = {
        0: "Sleep",
        1: "Bus Stop",
        2: "Person",
        3: "Road",
        4: "Bus",
        5: "Busdoor"
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
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        self.alertmode = 0 # SLEEP
        self.message = ""
        self.lastmodeclick = time.clock()
        self.resetFlags()
        self.tellMode()
        self.resetOrientation()

    def resetFlags(self):
        self.incomingsearch = 0
        self.followdoor = 0
        self.busdetection = 0
        self.busstopdetection = 0
        self.persondetection = 0

    def sayMessage(self):
        self.speak(self.message)

    def updateMessage(self, message):
        self.message = message

    def resetOrientation(self):
        self.hangle = 0
        self.vangle = 0
        self.sayMessage()

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
        elif self.alertmode==4:
            self.personStatus()

        self.iteration += 1

    def getAlertMode(self):
        return self.ALERT_MODES[self.alertmode]

    def __changeMode(self, increment=1):
        t = time.clock()
        diff = t - self.lastmodeclick 
        self.lastmodeclick = t
        if diff < 10:
            self.alertmode = (len(self.ALERT_MODES) + self.alertmode + increment)%len(self.ALERT_MODES)
            self.resetFlags()

        self.tellMode()

    def nextMode(self):
        self.__changeMode(1)

    def prevMode(self):
        self.__changeMode(-1)

    def tellMode(self):
        mode = self.getAlertMode()
        self.speak(f"{mode} mode")

    def speak(self, message):
        if message.strip():
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
            if nearestbus is not None:
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
                        if self.busdetection==0:
                            self.busdetection = 1
                            self.resetOrientation()
                        
                    # elif p["depth"] - c["depth"] < -0.8:
                    #     self.message = f"A bus is leaving"
                    elif abs(p["depth"] - c["depth"])<=0.5:
                        if self.busdetection<2:
                            self.busdetection = 2
                            self.message = f"A Bus is standing in front of you at less than {c['depth']*2} steps."
                            self.resetOrientation()

            else:
                self.message = "No bus found. Try facing the incoming direction."
                self.busdetection=0

    def busDoorStaus(self, bus):
        """
        From the nearest bus locate the door and tell about the direction
        """
        nearestbus = self.__nearestbus()
        if nearestbus is None:
            self.message = "No bus is there"
            return

        door = nearestbus.get(6, False) or nearestbus.get(8, False)
        if not door:
            self.message = "No door found. Turn a little left or right"
            if self.followdoor==1:
                self.sayMessage()
            self.followdoor = 0
        else:
            x = door['x']
            self.message = f"Door is in your front."
            if x<100:
                self.message = f"{self.message} Slightly turn left"
            elif x>100:
                self.message = f"{self.message} Slightly turn right"

            if not self.followdoor:
                self.resetOrientation()
                self.followdoor = 1

    #TODO: GPS based later
    def busStopStatus(self):
        """
        check for bus stops. May be we need to store the last bus stop found as well.
        """
        nearestbusstop = self.__nearestEntity("busstop")
        if nearestbusstop is not None:
            self.message = "There is a busstop in front of you"
            if self.busstopdetection==0:
                self.resetOrientation()
                self.busstopdetection = 1
        else:
            self.message = "No busstop found. Turn around slowly to locate."
            if self.busstopdetection==1:
                if abs(self.hangle)>10:
                    d = "left" if self.hangle<0 else "right"
                    self.message(f"turn a little {d} to find the bus stop.")
                self.sayMessage()
            self.busstopdetection = 0

    #TODO: compass and GPS can be used here later.
    def roadStatus(self):
        """
        check for road status, may be we need to remember the road and direct turn accordingly
        """

        if self.incomingsearch==3:
            if abs(self.hangle)>10:
                d = "left" if self.hangle<0 else "right"
                self.message(f"turn a little {d} to follow incoming bus direction.")
            return
            
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
            if not self.incomingsearch:
                self.message("Road found. Slowly turn right until no road detected.")
                self.incomingsearch = 1
                self.resetOrientation()
            elif self.incomingsearch==2:
                self.message("Stay in this direction. Switch to bus mode.")
                self.resetOrientation()
                self.incomingsearch = 3
            else:
                self.message("Slowly turn right until no road detected.")
        else:
            if self.incomingsearch == 1:
                self.message("Slowly turn left until road is found.")
                if self.incomingsearch==1:
                    self.sayMessage()
                    self.incomingsearch = 2
            else:
                self.message("You are not facing the road. Turn around slowly.")

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

            if not self.persondetection:
                self.resetOrientation()
                self.persondetection = 1
        else:
            self.message = "No person found. Turn around to check."
            self.persondetection = 0


        

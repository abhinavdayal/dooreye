from queue import Queue

def getDoorMesage(coords):
    horiz = "left" if coords.x<100 else "right" if coords.x>100 else None
    if horiz:
        return f"Bus door is located {round(coords.z/40)} steps to your front and about {round(coords.x/40)} steps to your {horiz}."
    else:
        return f"Bus door is located {round(coords.z/40)} steps to your front."

def getBusMessage(coords):
    pass

class AlertService:
    def __init__(self, fps = 10, duration = 60, gap = 1):
        self.iteration = 0
        self.fps = fps
        self.duration = duration
        self.gap = gap
        self.record = Queue(duration*fps)

    def process(self, objects):
        self.record.enqueue(objects)
        self.iteration += 1
        if self.iteration==1:
            return
        #print(objects["bus"])
        history = self.gap*self.fps
        if(self.record.size()>history): # gap duration passed
            prev = self.record.peek(-history)

            # check for bus
            cid = set(objects["bus"].keys())
            pid = set(prev["bus"].keys())
            commonids = cid.intersection(pid)

            for id in commonids:
                c = objects["bus"][id]["bus"]
                p = prev["bus"][id]["bus"]
                print(id, p["depth"], c["depth"])
                if p["depth"] - c["depth"] > 0.5:
                    print(f"bus {id} is approaching")
                elif p["depth"] - c["depth"] < -0.5:
                    print(f"bus {id} is leaving")
            # check for road
            # check for person
            # check for bus stop



        

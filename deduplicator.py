def area(roi):
    return (roi[2]-roi[0])*(roi[3]-roi[1])

def intersect(detections, i, j, threshold = 0.5):
    r1 = detections[i]
    r2 = detections[j]
    s1 = (r1["minx"], r1["miny"], r1["maxx"], r1["maxy"])
    s2 = (r2["minx"], r2["miny"], r2["maxx"], r2["maxy"])
    #print("S1, S2 = ", s1, s2)
    a1 = area(s1)
    a2 = area(s2)
    roi = ( max(s1[0], s2[0]), max(s1[1], s2[1]), min(s1[2], s2[2]), min(s1[3], s2[3]) )
    a3 = area(roi)
    
    #print("a1,a2,a3 = ", i, j, a1, a2, a3, a3/a1, a3/a2)
    if a3/a1 > threshold and a3/a2 > threshold:
        return i if a1>a2 else j
    else:
        return None

def obj_inside_bus(obj, bus):
    # try an overlap witha  certail threshold by subrtraction
    return obj["minx"]>=bus["minx"] and obj["miny"]>=bus["miny"] and obj["maxx"]<=bus["maxx"] and obj["maxy"]<=bus["maxy"]

def areatheshold(r1, threshold=100):
    return area((r1["minx"], r1["miny"], r1["maxx"], r1["maxy"])) >= threshold

def findunique(detections):
    l = len(detections)
    if l==1:
        return detections
    indexes = list(range(l))
    #print(l, indexes)
    for i in range(l):
        for j in range(i+1, l):
            if indexes[i] == indexes[j] or detections[i]["label"] != detections[j]["label"]:
                continue
            r = intersect(detections, i, j, 0.25) 
            if r is not None:
                #print(f"{i} and {j} are overlapping")
                indexes[j] = r
                indexes[i] = r
    #print(indexes)
    return [detections[i] for i in set(indexes)]


def FilterObjectsByBus(buses, objects):
    d = {b["id"]:{"bus": b} for b in buses}
    for o in objects:
        if o["label"] in [6,8,9]:
            for b in buses:
                if obj_inside_bus(o, b):
                    o["busid"] = b["id"]
                    r = d[b["id"]]
                    if o["label"] in r:
                        if o["confidence"] > r[o["label"]]["confidence"]: # if this is a beeter confidence
                            r[o["label"]] = o
                        elif o["confidence"] == r[o["label"]]["confidence"]: # or same confidence but better area
                            r1 = r[o["label"]]
                            a1 = area((r1["minx"], r1["miny"], r1["maxx"], r1["maxy"]))
                            a2 = area((o["minx"], o["miny"], o["maxx"], o["maxy"]))
                            if a2>a1:
                                r[o["label"]] = o
                    else:
                        r[o["label"]] = o
    return d

    # also remove multiple front / read doors / route in a bus and pick the largest one with most confidence



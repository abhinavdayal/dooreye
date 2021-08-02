def area(roi):
    return (roi[2]-roi[0])*(roi[3]-roi[1])

def intersect(detections, i, j, threshold = 0.5):
    r1 = detections[i]
    r2 = detections[j]
    s1 = (r1["roi"].topLeft().x, r1["roi"].topLeft().y, r1["roi"].bottomRight().x, r1["roi"].bottomRight().y)
    s2 = (r2["roi"].topLeft().x, r2["roi"].topLeft().y, r2["roi"].bottomRight().x, r2["roi"].bottomRight().y)
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
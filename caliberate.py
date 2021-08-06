import csv

distances = {
    "1.png":{ "view":"front", "distance":6},
    "2.png":{ "view":"front", "distance":9},
    "3.png":{ "view":"front", "distance":12},
    "4.png":{ "view":"side", "distance":6},
    "5.png":{ "view":"side", "distance":3},
    "6.png":{ "view":"diagonal", "distance":3},
    "7.png":{ "view":"diagonal", "distance":6},
    "8.png":{ "view":"diagonal", "distance":9},
    "9.png":{ "view":"front", "distance":3},
    }

H =  3#height of bus in meters

annotations = []
with open('snaps/annotations.csv', mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    for row in reader:
        annotations.append({header[i]:row[i] for i in range(len(row))})

reference = annotations[0]
D = distances[reference["filename"]]["distance"]
P = int(reference["h"])
F = (P*D)/H
print("F = ", F)
e = 0
per = 0
c = 0
for i in range(1, len(annotations)):
    a = annotations[i]
    if a["label"] != "bus":
        continue
    c += 1
    p = int(a["h"])
    d = (H*F)/p
    e += (distances[a['filename']]['distance'] - d)**2
    per += abs(distances[a['filename']]['distance'] - d)/d

    print(f"{a['filename']} [{distances[a['filename']]['view']}], Actual: {distances[a['filename']]['distance']}, Computed: {d}")

print("MSE =", e/c, "percent error =", per/c)

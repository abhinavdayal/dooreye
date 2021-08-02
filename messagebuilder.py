def getDoorMesage(coords):
    horiz = "left" if coords.x<100 else "right" if coords.x>100 else None
    if horiz:
        return f"Bus door is located {round(coords.z/40)} steps to your front and about {round(coords.x/40)} steps to your {horiz}."
    else:
        return f"Bus door is located {round(coords.z/40)} steps to your front."

def getBusMessage(coords):
    pass

from bluedot import BlueDot
from signal import pause

def onBluetoothConnect():
    print("Connected")

def onBluetoothDisconnect():
    print("Disconnected")

bd = BlueDot(print_messages=True)

bd.set_when_client_connects(onBluetoothConnect)
bd.set_when_client_disconnects(onBluetoothDisconnect)

def say_hello():
    print("Hello World")

def say_goodbye():
    print("goodbye")

bd.when_pressed = say_hello
bd.when_released = say_goodbye

pause()
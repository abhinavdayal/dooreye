from bluedot import BlueDot
from signal import pause
from logger import logger

def onBluetoothConnect():
    logger.info("Door Eye Armed")

def onBluetoothDisconnect():
    logger.info("Door Eye Disarmed")

bd = BlueDot(print_messages=True)

bd.set_when_client_connects(onBluetoothConnect)
bd.set_when_client_disconnects(onBluetoothDisconnect)

def say_hello():
    logger.info("Hello World")

def say_goodbye():
    logger.info("goodbye")

bd.when_pressed = say_hello
bd.when_released = say_goodbye

pause()
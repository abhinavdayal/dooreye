# Bus Door Detection

This project is to help the visually challenge board a bus by preciely directing them to the bus door. The project utilizes an OakD device running a custom mobilenet ssd model.

# install
0. [**Conditional**] On linux based systems add a new udev rule
```
$ echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
$ sudo udevadm control --reload-rules && sudo udevadm trigger
```
1. Create a python 3.7.* or 3.8.* virtual environment and activate it
2. Run `pip install -r requirements.txt`
3. Install Pico Text to Speech: `sudo apt-get install libttspico-utils`
4. Create a symlink for audio: `ln -s /dev/stdout speech.wav`
5. run `python tracker.py`
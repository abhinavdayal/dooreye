import os
import subprocess
#out = subprocess.Popen(['pactl', 'list', 'sinks'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#stdout, stderr = out.communicate()
#print("State: RUNNING" in str(stdout))

# out = subprocess.Popen(['pico2wave', '-w', 'speech.wav', '"This is a test"', '|', 'aplay'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# stdout, stderr = out.communicate()

cmd = f'pico2wave -w speech.wav "Hello there" | aplay'
os.system(cmd)
print("done")